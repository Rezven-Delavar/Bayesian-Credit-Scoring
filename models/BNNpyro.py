import torch
from torch import nn
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive, autoguide
from pyro import optim
import matplotlib.pyplot as plt
from Data_Loader import X_train_p, X_test_p, y_train , y_test



X_train_t = torch.tensor((X_train_p), dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor((X_test_p), dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)


class BNN(PyroModule):
    def __init__(self, in_size, hid_size=32, out_size=1):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](in_size, hid_size)
        self.fc2 = PyroModule[nn.Linear](hid_size, out_size)
        self.relu = nn.ReLU()
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hid_size, in_size]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hid_size]).to_event(1))
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([out_size, hid_size]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([out_size]).to_event(1))

    def forward(self, x, y=None):
        x = self.relu(self.fc1(x))
        logits = self.fc2(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            return pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

model = BNN(X_train_t.shape[1])
guide = autoguide.AutoDiagonalNormal(model)
svi = SVI(model, guide, optim.Adam({"lr": 0.01}), loss=Trace_ELBO())
pyro.clear_param_store()



print("Training BNN...")
for step in range(2000): 
    loss = svi.step(X_train_t, y_train_t)
    if step % 500 == 0: print(f"Step {step} Loss: {loss:.2f}")
print("Training finished.")


losses = []

for step in range(2000):

    loss = svi.step(X_train_t,y_train_t)
    losses.append(loss)

    if step%500==0:
        print("step:",step," loss:",loss)

# -------------------------
# Plot training loss
# -------------------------

plt.plot(losses)
plt.title("ELBO Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()


print("\n--- Prediction Results ---")
predictive = Predictive(model, guide=guide, num_samples=1000)
samples = predictive(X_test_t)
preds = samples['obs'].squeeze().numpy()
print(preds.shape)




###ELBO train vs. ELBO test###
train_losses = []
test_losses = []

for step in range(2000):
    train_loss = svi.step(X_train_t, y_train_t)
    train_losses.append(train_loss / len(X_train_t))  

    test_loss = svi.evaluate_loss(X_test_t, y_test_t)
    test_losses.append(test_loss / len(X_test_t))    

    if step % 500 == 0:
        print(f"Step {step} | Train ELBO: {train_losses[-1]:.4f} | Test ELBO: {test_losses[-1]:.4f}")
        
        

plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train ELBO Loss')
plt.plot(test_losses, label='Test ELBO Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('BNN Train/Test ELBO Loss')
plt.legend()
plt.grid(True)
plt.show()











customer_idx = 2
prob_good = preds[customer_idx].mean()
uncertainty = preds[customer_idx].std()
actual_status = 'Good' if y_test[customer_idx] == 1 else 'Bad'

print(f"Customer {customer_idx}: Actual Status = {actual_status}")
print(f"Predicted 'Good' Probability: {prob_good:.2%}")
print(f"Model Uncertainty: {uncertainty:.2%}")


##############ECE#############

p_pred = preds.mean(axis=0)

y_true = y_test


def expected_calibration_error(
    p_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 15,
    method: str = "binary",
) -> float:
    """ECE for binary classification.

    method="binary" (default): the bin's "accuracy" is the empirical positive
        rate; the bin's "confidence" is the mean of P(y=1).  This is the
        natural reliability for a probabilistic credit-scoring model.
    method="guo":  the Guo et al. (2017) definition for any number of classes
        in the binary special case.  Confidence = max(p, 1-p), accuracy =
        argmax(p)==y.  Useful for comparison against multi-class literature.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(p_pred)

    if method == "guo":
        pred = (p_pred >= 0.5).astype(int)
        conf = np.maximum(p_pred, 1.0 - p_pred)
        correct = (pred == y_true).astype(float)
        ece = 0.0
        for b in range(n_bins):
            lo, hi = bins[b], bins[b + 1]
            mask = (conf >= lo) & (conf < hi if b < n_bins - 1 else conf <= hi)
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / n) * abs(conf[mask].mean() - correct[mask].mean())
        return float(ece)

    # method == "binary"
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (p_pred >= lo) & (p_pred < hi if b < n_bins - 1 else p_pred <= hi)
        if mask.sum() == 0:
            continue
        conf = p_pred[mask].mean()
        pos_rate = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(conf - pos_rate)
    return float(ece)
ECE = expected_calibration_error(p_pred, y_true) 
print("ECE:",ECE)

def reliability_diagram_data(
    p_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> dict:
    """Return per-bin midpoints, mean P(y=1) (confidence), positive-rate
    (accuracy in the binary-reliability sense) and counts."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mids, confs, accs, counts = [], [], [], []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (p_pred >= lo) & (p_pred < hi if b < n_bins - 1 else p_pred <= hi)
        mids.append(0.5 * (lo + hi))
        if mask.sum() == 0:
            confs.append(np.nan)
            accs.append(np.nan)
            counts.append(0)
        else:
            confs.append(float(p_pred[mask].mean()))
            accs.append(float(y_true[mask].mean()))
            counts.append(int(mask.sum()))
    return {
        "bin_mid": np.array(mids),
        "confidence": np.array(confs),
        "accuracy": np.array(accs),
        "count": np.array(counts),
    }


def plot_reliability_diagram(p_pred, y_true, n_bins=10, title="Reliability Diagram BNN"):
    d = reliability_diagram_data(p_pred=p_pred, y_true=y_true, n_bins=n_bins)

    mids = d["bin_mid"]
    conf = d["confidence"]
    acc  = d["accuracy"]
    cnt  = d["count"]

    # فیلتر binهای خالی (NaN)
    m = ~np.isnan(conf) & ~np.isnan(acc) & (cnt > 0)
    mids, conf, acc, cnt = mids[m], conf[m], acc[m], cnt[m]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # --- Reliability diagram ---
    ax0 = ax
    ax0.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfectly calibrated")

    # پهنای میله‌ها ~ عرض bin
    bin_width = 1.0 / n_bins
    ax0.bar(mids, acc, width=bin_width*0.9, color = "#bcbd22", alpha=0.6, label="Empirical frequency ")
    ax0.bar(mids, conf, width=bin_width*0.9, color =  "#7f7f7f", alpha=0.6, label="predicted BNN")
    ax0.plot(mids, conf, "o-", color="black", markersize=4, lw=1, alpha=0.8, markerfacecolor="white", label="Mean predicted p")

    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.set_xlabel("Confidence")
    ax0.set_ylabel("Accuracy")
    ax0.set_title(title)
    ax0.legend(loc="best")

    plt.tight_layout()
    plt.show()

    return d 




print(p_pred.shape)
print(y_true.shape)

plot_reliability_diagram(p_pred=p_pred, y_true=y_true, n_bins=15)




