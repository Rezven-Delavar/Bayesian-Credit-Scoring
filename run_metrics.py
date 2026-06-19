# metrics the reviewer asked for (Accuracy/Precision/Recall/F1/AUC) alongside ECE,
# for BOTH models, on (a) the same 800/200 stratified split and (b) 5-fold CV.
import numpy as np, pandas as pd, torch, warnings
from torch import nn
import torch.optim as optim
import pyro, pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive, autoguide
from pyro import optim as pyopt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, brier_score_loss)
warnings.filterwarnings("ignore")

CSV = "german_credit.csv"
COLS = ["checking_status","duration_months","credit_history","purpose","credit_amount",
        "savings_status","employment_since","installment_rate","personal_status_sex",
        "other_parties","residence_since","property_magnitude","age_years",
        "other_payment_plans","housing","existing_credits","job","num_dependents",
        "telephone","foreign_worker","class"]
NUM = ["duration_months","credit_amount","installment_rate","residence_since",
       "age_years","existing_credits","num_dependents"]
CAT = [c for c in COLS[:-1] if c not in NUM]

data = pd.read_csv(CSV, header=0, names=COLS)
data["class"] = data["class"].apply(lambda x: 1.0 if x == 1 else 0.0).astype(int)  # 1=good, 0=bad/default
X = data.drop(columns=["class"]); y = data["class"].to_numpy()

def preprocess(Xtr, Xte):
    pp = ColumnTransformer([("num", StandardScaler(), NUM),
                            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
    return pp.fit_transform(Xtr).astype(np.float32), pp.transform(Xte).astype(np.float32)

# ---- the user's exact "binary" ECE (denominator = n, n_bins=15) ----
def ece_binary(p_pred, y_true, n_bins=15):
    bins = np.linspace(0,1,n_bins+1); n=len(p_pred); e=0.0
    for b in range(n_bins):
        lo,hi=bins[b],bins[b+1]
        m=(p_pred>=lo)&(p_pred< hi if b<n_bins-1 else p_pred<=hi)
        if m.sum()==0: continue
        e+=(m.sum()/n)*abs(p_pred[m].mean()-y_true[m].mean())
    return float(e)

# ---- BNN: identical to BNNpyro.py (1 hidden layer=32, N(0,1) priors, AutoDiagonalNormal, Adam 0.01) ----
class BNN(PyroModule):
    def __init__(self, in_size, hid=32, out=1):
        super().__init__()
        self.fc1=PyroModule[nn.Linear](in_size,hid); self.fc2=PyroModule[nn.Linear](hid,out); self.relu=nn.ReLU()
        self.fc1.weight=PyroSample(dist.Normal(0.,1.).expand([hid,in_size]).to_event(2))
        self.fc1.bias  =PyroSample(dist.Normal(0.,1.).expand([hid]).to_event(1))
        self.fc2.weight=PyroSample(dist.Normal(0.,1.).expand([out,hid]).to_event(2))
        self.fc2.bias  =PyroSample(dist.Normal(0.,1.).expand([out]).to_event(1))
    def forward(self,x,y=None):
        h=self.relu(self.fc1(x)); logits=self.fc2(h).squeeze(-1)
        with pyro.plate("data",x.shape[0]):
            return pyro.sample("obs",dist.Bernoulli(logits=logits),obs=y)

def run_bnn(Xtr,ytr,Xte,steps=3000,seed=0):
    pyro.clear_param_store(); pyro.set_rng_seed(seed); torch.manual_seed(seed)
    Xtr_t=torch.tensor(Xtr); ytr_t=torch.tensor(ytr,dtype=torch.float32); Xte_t=torch.tensor(Xte)
    m=BNN(Xtr_t.shape[1]); g=autoguide.AutoDiagonalNormal(m)
    svi=SVI(m,g,pyopt.Adam({"lr":0.01}),loss=Trace_ELBO())
    for _ in range(steps): svi.step(Xtr_t,ytr_t)
    pred=Predictive(m,guide=g,num_samples=1000)(Xte_t)["obs"].squeeze().numpy()
    return pred.mean(axis=0)  # P(y=1) per test point

# ---- ANN: identical to ANN.py (61->64->32->2, dropout .2, CE, Adam .001 wd 1e-3, 100 ep, bs20) ----
class ANN(nn.Module):
    def __init__(self,in_size):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_size,64),nn.ReLU(),nn.Dropout(0.2),
                               nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.2),nn.Linear(32,2))
    def forward(self,x): return self.net(x)

def run_ann(Xtr,ytr,Xte,epochs=100,seed=0):
    torch.manual_seed(seed)
    tr=torch.utils.data.TensorDataset(torch.tensor(Xtr),torch.tensor(ytr,dtype=torch.long))
    ld=torch.utils.data.DataLoader(tr,batch_size=20,shuffle=True)
    m=ANN(Xtr.shape[1]); crit=nn.CrossEntropyLoss(); opt=optim.Adam(m.parameters(),lr=1e-2,weight_decay=1e-4)
    m.train()
    for _ in range(epochs):
        for xb,yb in ld:
            opt.zero_grad(); loss=crit(m(xb),yb); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p=torch.softmax(m(torch.tensor(Xte)),dim=1)[:,1].numpy()
    return p

def metrics(p,y,thr=0.5):
    yhat=(p>=thr).astype(int)
    return dict(Acc=accuracy_score(y,yhat),
                Prec=precision_score(y,yhat,average="macro",zero_division=0),
                Rec=recall_score(y,yhat,average="macro",zero_division=0),
                F1=f1_score(y,yhat,average="macro",zero_division=0),
                AUC=roc_auc_score(y,p), Brier=brier_score_loss(y,p), ECE=ece_binary(p,y))

# ===== (A) single 800/200 stratified split (random_state=42), matching the paper =====
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
Xtr_p,Xte_p=preprocess(Xtr,Xte)
pB=run_bnn(Xtr_p,ytr,Xte_p); pA=run_ann(Xtr_p,ytr,Xte_p)
mB,mA=metrics(pB,yte),metrics(pA,yte)
keys=["Acc","Prec","Rec","F1","AUC","Brier","ECE"]
print("\n=== (A) Single 80/20 split (n_test=200) ===")
print(f"{'metric':6} | {'BNN':>7} | {'ANN':>7}")
for k in keys: print(f"{k:6} | {mB[k]:7.3f} | {mA[k]:7.3f}")

# ===== (B) 5-fold stratified CV (mean ± sd) =====
print("\n=== (B) 5-fold stratified CV (mean ± sd) ===")
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
accB={k:[] for k in keys}; accA={k:[] for k in keys}
for i,(tr_i,te_i) in enumerate(skf.split(X,y)):
    Xtr_p,Xte_p=preprocess(X.iloc[tr_i],X.iloc[te_i])
    ytr_,yte_=y[tr_i],y[te_i]
    pB=run_bnn(Xtr_p,ytr_,Xte_p,steps=2500,seed=i); pA=run_ann(Xtr_p,ytr_,Xte_p,seed=i)
    mb,ma=metrics(pB,yte_),metrics(pA,yte_)
    for k in keys: accB[k].append(mb[k]); accA[k].append(ma[k])
    print(f"  fold {i+1} done")
print(f"\n{'metric':6} | {'BNN mean±sd':>16} | {'ANN mean±sd':>16}")
for k in keys:
    print(f"{k:6} | {np.mean(accB[k]):6.3f} ± {np.std(accB[k]):5.3f} | {np.mean(accA[k]):6.3f} ± {np.std(accA[k]):5.3f}")
