import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_absolute_error ,mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from Data_Loader import X_train_p, X_test_p, y_train , y_test


class ANNDataset(Dataset):
    def __init__(self, features, labels ,transform):
        self.features = features
        self.labels = labels
        self.transform = transform
        
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx] 
        return x, y
       
    
#transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

#train_dataset = ANNDataset(X_train_p, y_train, transform=transform )
#test_dataset = ANNDataset(X_test_p, y_test, transform=transform)

train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_train_p, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)

test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_test_p, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)


train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)


for x, y in train_loader:
    break


x.shape, y.shape



def train(train_loader, model, criterion, optimizer, device):
  loss_per_epoch = []
  acc_per_epoch = []
  for data , label in train_loader:
    optimizer.zero_grad()
    data = data.to(device)
    label = label.to(device)
    pred = model(data)
    loss = criterion(pred, label)
    loss.backward()
    optimizer.step()
    loss_per_epoch.append(loss.item())
  return torch.mean(torch.tensor(loss_per_epoch))



def test(test_loader, model, criterion, device):
  loss_per_epoch = []
  acc_per_epoch = []
  with torch.no_grad():
    for data , label in test_loader:
      data = data.to(device)
      label = label.to(device)
      pred = model(data)
      loss = criterion(pred, label)
      loss_per_epoch.append(loss.item())
  return torch.mean(torch.tensor(loss_per_epoch))



class ANN(nn.Module):
    def __init__(self):
        super(ANN , self).__init__()
        self.fc1 = nn.Linear(61, 64)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        
        
        self.fc3 = nn.Linear(32,2)


    def forward(self , x):

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        
        return x
    
    
    
device = "cpu"
model = ANN().to(device)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 1e-4) #weight_decay = L2 

train_loss_per_epoch = []
test_loss_per_epoch =[]



for epoch in range(100):
  
  train_loss = train(train_loader, model, criterion, optimizer, device)
  train_loss_per_epoch.append(train_loss)
  
  test_loss = test(test_loader, model, criterion, device)
  test_loss_per_epoch.append(test_loss)

 
 
print(f"Best Test Loss: {min(test_loss_per_epoch):.4f}")
print(f"Best Train Loss: {min(train_loss_per_epoch):.4f}") 

##Train Loss vs.Test Loss##
plt.plot(range(len(train_loss_per_epoch)),train_loss_per_epoch, label="Train Loss")
plt.plot(range(len(test_loss_per_epoch)),test_loss_per_epoch, label="Test Loss")
plt.legend() 
plt.show()


##ECE##
model.eval()

all_probs = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)                     # خروجی خام مدل
        probs = torch.softmax(logits, dim=1)   # تبدیل به احتمال
        p_pred = probs[:, 1]                  # احتمال کلاس مثبت

        all_probs.append(p_pred.cpu())
        all_labels.append(y.cpu())

p_pred = torch.cat(all_probs).numpy()
y_true = torch.cat(all_labels).numpy()




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


def plot_reliability_diagram(p_pred, y_true, n_bins=10, title="Reliability Diagram ANN"):
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
    ax0.bar(mids, conf, width=bin_width*0.9, color =  "#7f7f7f", alpha=0.6, label="predicted ANN")
    ax0.plot(mids, conf, "o-", color="black", markersize=4, lw=1, alpha=0.8, markerfacecolor="white", label="Mean predicted p")

    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.set_xlabel("Confidence")
    ax0.set_ylabel("Accuracy")
    ax0.set_title(title)
    ax0.legend(loc="best")


    plt.tight_layout()
    plt.show()

    return d  # اگر خواستی بیرون هم استفاده کنی




print(p_pred.shape)
print(y_true.shape)


# p_pred: آرایه شکل (N,) احتمال کلاس 1  (مثلاً p_hat)
# y_true: آرایه شکل (N,) با 0/1
plot_reliability_diagram(p_pred=p_pred, y_true=y_true, n_bins=15)

