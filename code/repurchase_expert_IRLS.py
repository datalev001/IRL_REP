# -*- coding: utf-8 -*-
"""
E-commerce Promotion & Repeat (Scenario A) · IRL/GRPO training (data-only, no generator)
----------------------------------------------------------------------------
Inputs (must already exist):
  - C:/pj/promo_traj.csv
  - C:/pj/context_daily.csv

Outputs (all saved to the same directory C:/pj):
  - irlA_w.npy, irlA_scaler.npz
  - irlB_w.npy, irlB_scaler.npz
  - grpo_actor.pt
  - ope_snips.csv
  - today_offer_recos.csv
  - exec_summary.txt
  - Several visualization PNGs (weights, confusion matrix, training curve, OPE comparison, daily recommendation mix)
"""

from pathlib import Path
import os, json, math
import numpy as np
import pandas as pd

# Use a non-interactive matplotlib backend so it runs in any environment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# ----------------------------- Config -----------------------------
DATA_DIR   = Path("C:/pj")   # single path for both inputs and outputs
PROMOS     = ["C5","C10","FS","BDL","LP","MSG"]
VAL_SPLIT  = 0.85
RNG_SEED   = 2025
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

# Business cost parameters (must match the ones used during data generation)
OP_COST      = {"HOLD":0.0, "PROMO":0.03}
COUPON_COST  = {"C5":5.0, "C10":10.0}
SHIP_COST    = 8.0
POINTS_COST  = 0.01
VARIANT_COST = {"C5":5.0, "C10":10.0, "FS":SHIP_COST, "BDL":0.0, "LP":3.0, "MSG":0.0}

# OPE reward: r = profit_1d + ALPHA_REP*repurchase_30d − DELTA_FATIGUE*fatigue
ALPHA_REP     = 0.5
DELTA_FATIGUE = 0.2

# GRPO (critic-free, group-relative)
EPOCHS_GRPO = 25
LR_GRPO     = 2e-4
TAU_LIST    = 0.8          # actor softmax temperature
TAU_REF     = 0.9          # reference softmax temperature (derived from rewards)
BETA_KL     = 0.2          # KL(π||π_ref) weight
COST_COEF   = 0.05
ENT_COEF    = 0.01

# ------------------------- Utilities -------------------------
def ensure_inputs():
    a = DATA_DIR / "promo_traj.csv"
    b = DATA_DIR / "context_daily.csv"
    if not a.exists() or not b.exists():
        missing = []
        if not a.exists(): missing.append(str(a))
        if not b.exists(): missing.append(str(b))
        raise FileNotFoundError(
            "noCSV，place ：\n  - " + "\n  - ".join(missing)
        )

def log1p_scale(x):      # for recency_days
    return np.log1p(np.maximum(0.0, x))

def softmax_row(z, tau=1.0):
    z = z / max(tau, 1e-6)
    m = np.max(z)
    e = np.exp(np.clip(z - m, -50, 50))
    return e / (np.sum(e) + 1e-12)

def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

# ------------------------- Load data -------------------------
ensure_inputs()
traj = pd.read_csv(DATA_DIR / "promo_traj.csv", parse_dates=["date"])

need_cols = [
    "date","session_id","expert_id","user_id","category","channel",
    "action_type","ticker","notional","fee","profit_1d","repurchase_30d","conv_1d",
    "price_sens","disc_aff","engage","churn","cat_aff","inv_idx","margin_rate","seasonality",
    "fatigue","comp_price_idx","coupon_elig","recency_days",
    "ret60_lag1","price_vs_ma20_lag1","vol20_lag1"
]
for c in need_cols:
    if c not in traj.columns:
        raise RuntimeError(f"promo_traj.csv no columns: {c}")

traj = traj[need_cols].copy().sort_values("date").reset_index(drop=True)

# ------------------------- Optional context (for AOV lag lookup) -------------------------
_ctx_idx = None
_ctx_path = DATA_DIR / "context_daily.csv"
if _ctx_path.exists():
    ctx_df = pd.read_csv(_ctx_path, parse_dates=["date"])
    _ctx_idx = ctx_df.set_index(["date","category"])
else:
    print("[WARN] Not finding context_daily.csv，AOV using default: 150.")

def _get_ctx_aov_lag(row):
    aov_lag = 150.0
    if _ctx_idx is not None:
        key = (row["date"], row["category"])
        if key in _ctx_idx.index:
            rec = _ctx_idx.loc[key]
            aov_lag = float(rec["aov_base_lag1"]) if "aov_base_lag1" in rec and pd.notna(rec["aov_base_lag1"]) \
                      else float(rec.get("aov_base", 150.0))
    return aov_lag

# ------------------------- Time split -------------------------
dates_all = traj["date"].unique()
cut_idx   = int(len(dates_all) * VAL_SPLIT)
train_dates = set(dates_all[:cut_idx])
val_dates   = set(dates_all[cut_idx:])
VAL_START   = pd.to_datetime(dates_all[cut_idx]) if cut_idx < len(dates_all) else traj["date"].max()
print(f"[split] train_days={len(train_dates)}  val_days={len(val_dates)}  val_start={VAL_START.date()}")

# ------------------------- Candidate rule set -------------------------
def feasible_variants(row):
    inv_idx = float(row["inv_idx"])
    elig    = int(row["coupon_elig"])
    cand = PROMOS[:]
    if inv_idx < 0.15:
        cand = [x for x in cand if x not in ("C10","BDL")]
    if not elig:
        cand = [x for x in cand if x not in ("C10",)]
    if len(cand) == 0:
        cand = ["C5","MSG"]
    return cand

# ------------------------- Stage-A: IRL (binary classification) -------------------------
A_cols_raw = [
    "price_sens","disc_aff","engage","churn","cat_aff","inv_idx","margin_rate","seasonality",
    "fatigue","comp_price_idx","coupon_elig",
    "ret60_lag1","price_vs_ma20_lag1","vol20_lag1"
]
def build_A_matrix(df):
    X = df[A_cols_raw].astype(float).values
    bias = np.ones((X.shape[0],1), dtype=np.float64)
    X = np.hstack([X, bias])
    idx_bias = X.shape[1]-1
    mean = X[:,:idx_bias].mean(axis=0)
    std  = X[:,:idx_bias].std(axis=0) + 1e-6
    X[:,:idx_bias] = (X[:,:idx_bias] - mean) / std
    scaler = dict(mean=mean, std=std, idx_bias=idx_bias, cols=A_cols_raw)
    return X.astype(np.float64), scaler

def stageA_train():
    df = traj.copy()
    df["y"] = (df["action_type"]=="PROMO").astype(int)
    mtr = df["date"].isin(train_dates).values
    mva = df["date"].isin(val_dates).values

    XA, scalerA = build_A_matrix(df)
    yA = df["y"].values.astype(np.int64)

    pos, neg = (yA==1).sum(), (yA==0).sum()
    w_pos = len(yA) / (2.0*max(1,pos))
    w_neg = len(yA) / (2.0*max(1,neg))
    wA = np.where(yA==1, w_pos, w_neg).astype(np.float64)

    def train_logistic(X, y, ws, lr=0.08, reg=2e-3, epochs=200):
        w = np.zeros(X.shape[1], dtype=np.float64)
        for ep in range(1, epochs+1):
            p = sigmoid(X @ w)
            grad = (ws * (y - p)) @ X - reg*w
            w += lr * grad / max(1, X.shape[0])
            if ep % 40 == 0:
                nll = -np.mean(ws*(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12)))
                print(f"[IRL-A] ep {ep:03d} | wNLL={nll:.4f} | ||w||={np.linalg.norm(w):.3f}")
        return w

    wA_star = train_logistic(XA[mtr], yA[mtr], wA[mtr])

    def eval_A(X, y, w):
        p = sigmoid(X @ w); pred = (p>=0.5).astype(int)
        tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum()); fn = int(((pred==0)&(y==1)).sum())
        acc = float((pred==y).mean())
        prec= tp/max(1,tp+fp); rec = tp/max(1,tp+fn); f1 = 2*prec*rec/max(1e-9,prec+rec)
        nll = -np.mean(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12))
        return dict(n=len(y), acc=acc, precision=prec, recall=rec, f1=f1, nll=nll)

    print("\n[Stage-A Eval]")
    print("  train:", eval_A(XA[mtr], yA[mtr], wA_star))
    print("    val:", eval_A(XA[mva], yA[mva], wA_star))

    # Save weights & scaler
    np.save(DATA_DIR/"irlA_w.npy", wA_star)
    np.savez(DATA_DIR/"irlA_scaler.npz", **scalerA)

    # Weight bar chart
    names = A_cols_raw + ["bias"]
    plt.figure(figsize=(7.5,3.2))
    plt.bar(range(len(wA_star)), wA_star)
    plt.xticks(range(len(wA_star)), names, rotation=30, ha="right")
    plt.title("Stage-A Feature Weights (PROMO vs HOLD)")
    plt.tight_layout()
    plt.savefig(DATA_DIR/"plot_stageA_weights.png", dpi=150)
    plt.close()

    # Confusion matrix on validation
    p = sigmoid(XA[mva] @ wA_star); pred=(p>=0.5).astype(int); y=yA[mva]
    cm = np.zeros((2,2), dtype=int)
    for yt, yp in zip(y, pred): cm[int(yt),int(yp)] += 1
    fig, ax = plt.subplots(figsize=(3.8,3.2))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j,i,cm[i,j],ha="center",va="center")
    ax.set_xticks([0,1]); ax.set_xticklabels(["HOLD","PROMO"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["HOLD","PROMO"])
    ax.set_xlabel("Pred"); ax.set_ylabel("Actual"); ax.set_title("Stage-A Confusion (Val)")
    plt.tight_layout()
    plt.savefig(DATA_DIR/"plot_stageA_confusion_val.png", dpi=150)
    plt.close()

    return wA_star, scalerA

wA_star, scalerA = stageA_train()

# ------------------------- Stage-B: IRL (multiclass softmax) -------------------------
B_base_cols = [
    "price_sens","disc_aff","engage","churn","cat_aff","inv_idx","margin_rate","seasonality",
    "fatigue","comp_price_idx","coupon_elig","recency_days",
    "ret60_lag1","price_vs_ma20_lag1","vol20_lag1"
]
INTER_NAMES = [
    "ps_x_coupon",     # price_sens × 1[coupon]
    "inv_x_bundle",    # inv_idx × 1[BDL]
    "elig_x_lp",       # coupon_elig × 1[LP]
    "churn_x_msg",     # churn × 1[MSG]
    "aovproxy_x_fs"    # ret60_lag1 × 1[FS] (AOV proxy)
]

def build_B_matrix_and_scaler(df):
    Xb = df[B_base_cols].astype(float).values.copy()
    # recency_days -> log1p
    idx_rec = B_base_cols.index("recency_days")
    Xb[:, idx_rec] = log1p_scale(Xb[:, idx_rec])
    mean = Xb.mean(axis=0)
    std  = Xb.std(axis=0) + 1e-6
    idx_bias = len(B_base_cols)  # base z-scored, then append one-hot + interactions + bias
    scaler = dict(mean=mean, std=std, idx_bias=idx_bias, cols=B_base_cols)
    return scaler

scalerB = build_B_matrix_and_scaler(traj)
np.savez(DATA_DIR/"irlB_scaler.npz", **scalerB)

def build_B_feature_row(row, variant, scaler=None):
    base = row[B_base_cols].astype(float).values.copy()
    base[B_base_cols.index("recency_days")] = log1p_scale(base[B_base_cols.index("recency_days")])
    base = base.astype(np.float64)
    if scaler is not None:
        mu, sd, idx_bias = scaler["mean"], scaler["std"], scaler["idx_bias"]
        base[:idx_bias] = (base[:idx_bias] - mu) / sd

    onehot = np.zeros(len(PROMOS), dtype=np.float64)
    if variant in PROMOS: onehot[PROMOS.index(variant)] = 1.0

    # Variant-dependent interactions
    is_coupon = 1.0 if variant in ("C5","C10") else 0.0
    is_bdl    = 1.0 if variant=="BDL" else 0.0
    is_lp     = 1.0 if variant=="LP" else 0.0
    is_msg    = 1.0 if variant=="MSG" else 0.0
    is_fs     = 1.0 if variant=="FS"  else 0.0

    ps   = float(row["price_sens"])
    inv  = float(row["inv_idx"])
    elig = float(row["coupon_elig"])
    churn= float(row["churn"])
    aovp = float(row["ret60_lag1"])

    inter = np.array([
        ps * is_coupon,
        inv * is_bdl,
        elig * is_lp,
        churn * is_msg,
        aovp * is_fs
    ], dtype=np.float64)

    feat = np.hstack([base, onehot, inter, np.array([1.0], dtype=np.float64)])
    return feat

def stageB_build_sets():
    tr_list, tr_y, va_list, va_y = [], [], [], []
    for _, row in traj.iterrows():
        if row["action_type"] != "PROMO": continue
        cands = feasible_variants(row)
        if len(cands)==0: continue
        if not isinstance(row["ticker"], str) or row["ticker"] not in cands: continue
        # Put the logged expert's choice at index 0
        cands = [row["ticker"]] + [x for x in cands if x != row["ticker"]]
        Xk = np.vstack([build_B_feature_row(row, v, scaler=scalerB) for v in cands])
        if row["date"] in train_dates:
            tr_list.append(Xk); tr_y.append(0)
        elif row["date"] in val_dates:
            va_list.append(Xk); va_y.append(0)
    return tr_list, np.array(tr_y, np.int64), va_list, np.array(va_y, np.int64)

X_B_tr, y_B_tr, X_B_va, y_B_va = stageB_build_sets()
feat_dim = (X_B_tr[0].shape[1] if X_B_tr else 0)
print(f"[Stage-B] train groups={len(X_B_tr)}  val groups={len(X_B_va)}  feature_dim={feat_dim}")

def stageB_train_softmax(X_list, y, lr=0.20, reg=2e-3, epochs=80, tau_sched=(1.2,0.9,0.8)):
    if not X_list: return np.zeros(feat_dim, dtype=np.float64)
    w = np.zeros(X_list[0].shape[1], dtype=np.float64)
    b1, b2 = epochs//3, 2*epochs//3
    for ep in range(1, epochs+1):
        tau = tau_sched[0] if ep<=b1 else (tau_sched[1] if ep<=b2 else tau_sched[2])
        idx = np.random.permutation(len(X_list))
        ll, n = 0.0, 0
        grad = np.zeros_like(w)
        for j in idx:
            Xk = X_list[j]; yk = y[j]
            p  = softmax_row(Xk @ w, tau=tau)
            ll += np.log(p[yk] + 1e-12)
            grad += Xk[yk] - (p[:,None]*Xk).sum(axis=0)
            n += 1
        grad -= reg * w
        w += lr * grad / max(1, n)
        if ep % 20 == 0:
            print(f"[IRL-B] ep {ep:03d} | avg NLL={-ll/max(1,n):.4f} | ||w||={np.linalg.norm(w):.3f} | tau={tau}")
    return w

wB_star = stageB_train_softmax(X_B_tr, y_B_tr)
np.save(DATA_DIR/"irlB_w.npy", wB_star)

def top1_acc(X_list, y, w, tau=0.8):
    hit = 0
    for Xk, yk in zip(X_list, y):
        p = softmax_row(Xk @ w, tau=tau)
        if np.argmax(p)==int(yk): hit += 1
    return hit/max(1,len(y))

def ndcg_at_k(X_list, y, w, k=3, tau=0.8):
    tot = 0.0
    for Xk, yk in zip(X_list, y):
        p = softmax_row(Xk @ w, tau=tau)
        order = np.argsort(-p)
        rank = int(np.where(order==yk)[0][0]) + 1
        dcg = 1.0 / math.log2(rank+1)
        tot += dcg
    return tot / max(1,len(y))

print("\n[Stage-B Eval]")
print("  train top1:", top1_acc(X_B_tr, y_B_tr, wB_star, tau=0.8))
print("    val top1:", top1_acc(X_B_va, y_B_va, wB_star, tau=0.8))
print("  train NDCG@3:", ndcg_at_k(X_B_tr, y_B_tr, wB_star, k=3, tau=0.8))
print("    val NDCG@3:", ndcg_at_k(X_B_va, y_B_va, wB_star, k=3, tau=0.8))

# Stage-B weight plot (includes one-hot & interactions)
namesB = B_base_cols + [f"1hot_{v}" for v in PROMOS] + INTER_NAMES + ["bias"]
plt.figure(figsize=(10.5,3.2))
plt.bar(range(len(wB_star)), wB_star)
plt.xticks(range(len(wB_star)), namesB, rotation=35, ha="right")
plt.title("Stage-B Feature Weights (Which Promo, with interactions)")
plt.tight_layout()
plt.savefig(DATA_DIR/"plot_stageB_weights.png", dpi=150)
plt.close()

# ------------------------- GRPO (critic-free, group-relative) -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GRPOActor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, Xk):        # Xk: (K,D)
        return self.net(Xk).squeeze(-1)  # (K,)

def softmax_t(x, tau):
    x = x / max(tau,1e-6)
    m = x.max()
    e = torch.exp(torch.clamp(x-m, -50, 50))
    return e / (e.sum() + 1e-12)

# Deterministic business reward estimator (aligned with generator logic)
def _expected_reward_for_variant(row, variant):
    ps   = float(row.get("price_sens", 0.5))
    da   = float(row.get("disc_aff", 0.5))
    engage = float(row.get("engage", 0.5))
    churn  = float(row.get("churn", 0.3))
    cat_aff= float(row.get("cat_aff", 0.3))
    inv    = float(row.get("inv_idx", 0.5))
    margin = float(row.get("margin_rate", 0.3))
    cpi    = float(row.get("comp_price_idx", 1.0))
    seas   = float(row.get("seasonality", 1.0))
    fatigue= float(row.get("fatigue", 0.0))
    coupon_elig = int(row.get("coupon_elig", 0))
    ret60_proxy = float(row.get("ret60_lag1", 0.04))

    aov_lag = _get_ctx_aov_lag(row)

    base_conv = np.clip( (0.6*ret60_proxy) * (0.7 + 0.2*seas) * (0.7 + 0.6*cat_aff), 0.001, 0.5 )

    uplift = 0.0
    if variant == "C10": uplift = 0.05 + 0.12*ps + 0.04*da
    elif variant == "C5":  uplift = 0.03 + 0.08*ps + 0.03*da
    elif variant == "FS":  uplift = 0.04 + 0.06*(aov_lag<150) + 0.03*churn
    elif variant == "BDL": uplift = 0.02 + 0.10*cat_aff + 0.04*engage
    elif variant == "LP":  uplift = 0.015 + 0.02*engage + 0.02*coupon_elig
    elif variant == "MSG": uplift = 0.02 + 0.01*engage
    uplift = max(0.0, uplift - 0.04*fatigue - 0.03*(inv < 0.15))

    p_conv = float(np.clip(base_conv + uplift, 0.001, 0.9))

    aov_exp = aov_lag * (1.0 + (0.10 if variant=="BDL" else 0.0)
                         - (0.06 if cpi<0.9 else 0.0) - (0.05 if inv<0.12 else 0.0))

    promo_cost = 0.0
    if variant in ("C5","C10"): promo_cost = COUPON_COST[variant]
    elif variant == "FS":       promo_cost = SHIP_COST
    elif variant == "LP":       promo_cost = POINTS_COST * aov_exp
    exp_cost = OP_COST["PROMO"] + p_conv * promo_cost
    exp_gross= margin * aov_exp * p_conv
    exp_profit = exp_gross - exp_cost

    p_rep = float(np.clip(0.08 + 0.15*engage + 0.06*p_conv + 0.05*coupon_elig
                          - 0.04*(variant in ("C10","C5") and ps>0.8), 0.01, 0.8))
    reward = exp_profit + ALPHA_REP * p_rep - DELTA_FATIGUE * fatigue
    return float(reward)

def reward_vector_for_group(row, cands):
    return np.array([_expected_reward_for_variant(row, v) for v in cands], dtype=np.float32)

def build_A_matrix_full(df):
    X, sc = build_A_matrix(df)  # reuse Stage-A scaler
    return X, sc

def build_grpo_groups(df, gate=0.50):
    groups = []
    XA_all, _ = build_A_matrix_full(df)
    def p_A(i): return float(sigmoid(XA_all[i] @ wA_star))
    for i, row in df.iterrows():
        if row["date"] not in train_dates: 
            continue
        cands = feasible_variants(row)
        if len(cands)==0: continue
        if p_A(i) < gate: continue
        Xk = np.vstack([build_B_feature_row(row, v, scaler=scalerB) for v in cands]).astype(np.float32)
        r  = reward_vector_for_group(row, cands).astype(np.float32)
        cost = np.array([VARIANT_COST.get(v, 0.0) + OP_COST["PROMO"] for v in cands], dtype=np.float32)
        pref_ref = softmax_row(r, tau=TAU_REF).astype(np.float32)
        groups.append(dict(X=Xk, reward=r, pref_ref=pref_ref, cost=cost, cands=cands, date=row["date"]))
    return groups

grpo_tr_groups = build_grpo_groups(traj, gate=0.50)

# Validation groups (use the same gate)
grpo_va_groups = []
XA_all, _ = build_A_matrix_full(traj)
def p_A_idx(i): return float(sigmoid(XA_all[i] @ wA_star))
for i, row in traj.iterrows():
    if row["date"] not in val_dates: continue
    cands = feasible_variants(row)
    if len(cands)==0: continue
    if p_A_idx(i) < 0.50: continue
    Xk = np.vstack([build_B_feature_row(row, v, scaler=scalerB) for v in cands]).astype(np.float32)
    r  = reward_vector_for_group(row, cands).astype(np.float32)
    cost = np.array([VARIANT_COST.get(v, 0.0) + OP_COST["PROMO"] for v in cands], dtype=np.float32)
    pref_ref = softmax_row(r, tau=TAU_REF).astype(np.float32)
    grpo_va_groups.append(dict(X=Xk, reward=r, pref_ref=pref_ref, cost=cost, cands=cands, date=row["date"]))

print(f"[GRPO] train groups={len(grpo_tr_groups)}  val groups={len(grpo_va_groups)}")

D_B = len(B_base_cols) + len(PROMOS) + len(INTER_NAMES) + 1
actor = GRPOActor(in_dim=D_B).to(DEVICE)
opt   = torch.optim.Adam(actor.parameters(), lr=LR_GRPO, weight_decay=1e-6)

def grpo_epoch(groups, train=True):
    (actor.train() if train else actor.eval())
    total = 0.0
    for g in groups:
        X = torch.tensor(g["X"], dtype=torch.float32, device=DEVICE)           # (K,D)
        r = torch.tensor(g["reward"], dtype=torch.float32, device=DEVICE)      # (K,)
        c = torch.tensor(g["cost"],   dtype=torch.float32, device=DEVICE)      # (K,)
        p_ref = torch.tensor(g["pref_ref"], dtype=torch.float32, device=DEVICE)# (K,)
        with torch.set_grad_enabled(train):
            logits = actor(X)                                                  # (K,)
            p = softmax_t(logits, TAU_LIST)                                    # (K,)

            # Group-relative advantage (zero mean, normalized)
            r_mean = r.mean()
            r_std  = torch.clamp(r.std(unbiased=False), min=1e-3)
            A = (r - r_mean) / r_std

            eps = 1e-12
            # KL to reference policy π_ref(r)
            kl = torch.sum(p * (torch.log(p + eps) - torch.log(p_ref + eps)))

            # Policy gradient objective
            loss_pg = -(A.detach() * torch.log(p + eps)).sum()
            loss_cost = (p * c).sum()
            ent = -(p * torch.log(p + eps)).sum()

            loss = loss_pg + BETA_KL*kl + COST_COEF*loss_cost - ENT_COEF*ent

            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        total += float(loss.item())
    return total / max(1, len(groups))

train_curve = []
for ep in range(1, EPOCHS_GRPO+1):
    loss_tr = grpo_epoch(grpo_tr_groups, train=True)
    loss_va = grpo_epoch(grpo_va_groups, train=False)
    train_curve.append((ep, loss_tr, loss_va))
    if ep % 5 == 0:
        print(f"[GRPO] epoch {ep:02d} | train={loss_tr:.4f} | val={loss_va:.4f}")

# Save actor weights
torch.save(actor.state_dict(), DATA_DIR/"grpo_actor.pt")

# Plot training curve
ep_axis, tr_vals, va_vals = zip(*train_curve)
plt.figure(figsize=(6.5,3.2))
plt.plot(ep_axis, tr_vals, label="train")
plt.plot(ep_axis, va_vals, label="val")
plt.xlabel("epoch"); plt.ylabel("loss")
plt.title("GRPO Training Curve (critic-free, group-relative)")
plt.legend(); plt.tight_layout()
plt.savefig(DATA_DIR/"plot_grpo_curve.png", dpi=150)
plt.close()

# ------------------------- OPE: SNIPS (validation window) -------------------------
def pi_b_IRLB(row, cands, tau=0.8):
    Xk = np.vstack([build_B_feature_row(row, v, scaler=scalerB) for v in cands])
    z  = Xk @ wB_star
    return softmax_row(z, tau=tau)

def pi_e_actor(row, cands):
    Xk = np.vstack([build_B_feature_row(row, v, scaler=scalerB) for v in cands]).astype(np.float32)
    with torch.no_grad():
        logits = actor(torch.tensor(Xk, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    z = logits / max(TAU_LIST, 1e-6); z = z - np.max(z)
    e = np.exp(np.clip(z, -50, 50))
    return e / (e.sum()+1e-12)

def row_reward(row):
    return float(row["profit_1d"]) + ALPHA_REP*float(row["repurchase_30d"]) - DELTA_FATIGUE*float(row["fatigue"])

ope_logs = []
val_df = traj[traj["date"].isin(val_dates)].reset_index(drop=True)
for _, row in val_df.iterrows():
    if row["action_type"] != "PROMO": continue
    cands = feasible_variants(row)
    if len(cands)==0 or not (isinstance(row["ticker"], str) and row["ticker"] in cands): continue
    pb = pi_b_IRLB(row, cands, tau=0.8)
    pe = pi_e_actor(row, cands)
    aidx = cands.index(row["ticker"])
    r    = row_reward(row)
    ope_logs.append(dict(date=str(row["date"].date()), category=row["category"], action=row["ticker"],
                         r=r, w=pe[aidx]/max(pb[aidx],1e-8)))

ope_df = pd.DataFrame(ope_logs)
if len(ope_df)==0:
    print("[OPE] No PROMO in validation set")
else:
    cap = 50.0
    w = np.clip(ope_df["w"].values.astype(np.float64), 0.0, cap)
    r = ope_df["r"].values.astype(np.float64)
    ips   = float(np.mean(w * r))
    snips = float((w * r).sum() / (w.sum() + 1e-12))

    baseline = float(r.mean())
    delta_snips = float(snips - baseline)

    # Bootstrap CI
    BOOT = 300
    rng = np.random.default_rng(RNG_SEED)
    bs = []
    n  = len(ope_df)
    for _ in range(BOOT):
        idx = rng.integers(0, n, size=n)
        wb, rb = w[idx], r[idx]
        bs.append(float((wb*rb).sum() / (wb.sum()+1e-12)))
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    # Save table
    ope_df.to_csv(DATA_DIR/"ope_snips.csv", index=False)

    # Visualization
    fig, ax = plt.subplots(figsize=(6.0,3.2))
    ax.bar([0,1], [baseline, snips], width=0.45, tick_label=["Logged Baseline","SNIPS (Pred)"])
    ax.errorbar([1], [snips], yerr=[[snips-lo],[hi-snips]], fmt='o', capsize=4, label="95% CI")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"OPE - SNIPS vs Baseline  (Δ={delta_snips:.3f}, N={len(ope_df)})")
    ax.legend(); plt.tight_layout()
    plt.savefig(DATA_DIR/"plot_ope_snips.png", dpi=150)
    plt.close()

    print(json.dumps({
        "OPE":{
            "N": int(len(ope_df)),
            "BaselineLogged": round(baseline,6),
            "IPS": round(ips,6),
            "SNIPS": round(snips,6),
            "Delta_SNIPS_vs_Baseline": round(delta_snips,6),
            "SNIPS_CI95":[round(lo,6), round(hi,6)],
            "Cap": cap
        }
    }, ensure_ascii=False, indent=2))

# ------------------------- Demo: today's top offers + explanations -------------------------
def stageA_prob_for_row(row):
    x = row[A_cols_raw].astype(float).values.copy()
    x = np.hstack([x, np.array([1.0])])
    idx_bias = scalerA["idx_bias"]
    x[:idx_bias] = (x[:idx_bias]-scalerA["mean"])/scalerA["std"]
    return float(sigmoid(x @ wA_star))

def explain_variant_scores(row, cands, topk=3):
    Xk = np.vstack([build_B_feature_row(row, v, scaler=scalerB) for v in cands])
    with torch.no_grad():
        logits = actor(torch.tensor(Xk, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    z = logits / max(TAU_LIST, 1e-6); z = z - np.max(z)
    p = np.exp(np.clip(z, -50, 50)); p = p / (p.sum()+1e-12)
    order = np.argsort(-p)[:min(topk, len(cands))]
    reasons = []
    for i in order:
        feat = Xk[i]
        score_irl = float(feat @ wB_star)
        reasons.append(dict(variant=cands[i], policy_score=float(p[i]), irl_score=score_irl))
    return reasons

last_day = traj["date"].max()
subset = traj[traj["date"]==last_day]
demo_rows = subset.sample(n=min(500, len(subset)), random_state=RNG_SEED) if len(subset)>0 else subset
reco = []
for _, row in demo_rows.iterrows():
    pA = stageA_prob_for_row(row)
    if pA < 0.5: continue
    cands = feasible_variants(row)
    if len(cands)==0: continue
    reasons = explain_variant_scores(row, cands, topk=3)
    for r in reasons:
        reco.append({
            "date": str(row["date"].date()),
            "session_id": int(row["session_id"]),
            "category": row["category"],
            "pA_PROMO": round(pA,4),
            "variant": r["variant"],
            "policy_score": round(r["policy_score"],6),
            "irl_score": round(r["irl_score"],6)
        })

if len(reco)>0:
    df_reco = pd.DataFrame(reco).sort_values(["policy_score"], ascending=False)
    df_reco.to_csv(DATA_DIR/"today_offer_recos.csv", index=False)
    cnt = df_reco["variant"].value_counts().reindex(PROMOS, fill_value=0)
    plt.figure(figsize=(5.5,3.2))
    plt.bar(cnt.index, cnt.values)
    plt.title(f"Today Reco Variant Mix - {str(last_day.date())}")
    plt.tight_layout()
    plt.savefig(DATA_DIR/"plot_today_reco_mix.png", dpi=150)
    plt.close()
    print(f"[Demo] Saved: today_offer_recos.csv  (rows={len(df_reco)})")
else:
    print("[Demo] no last day sample: A-head PROMO gate 。")

# ------------------------- Executive summary -------------------------
lines = []
lines.append(f"- Train/Val split: {len(train_dates)} / {len(val_dates)} days (val_start={VAL_START.date()}).")
if (DATA_DIR/"irlA_w.npy").exists():
    lines.append("- IRL-A (PROMO vs HOLD) trained; weights & plots saved.")

if (DATA_DIR/"irlB_w.npy").exists():
    try:
        top1_tr_val = top1_acc(X_B_tr, y_B_tr, wB_star, tau=0.8) if len(X_B_tr) > 0 else float('nan')
        top1_va_val = top1_acc(X_B_va, y_B_va, wB_star, tau=0.8) if len(X_B_va) > 0 else float('nan')
        ndcg_tr = ndcg_at_k(X_B_tr, y_B_tr, wB_star, 3, 0.8) if len(X_B_tr) > 0 else float('nan')
        ndcg_va = ndcg_at_k(X_B_va, y_B_va, wB_star, 3, 0.8) if len(X_B_va) > 0 else float('nan')
        lines.append(
            f"- IRL-B (Which Promo) top1 train/val: {top1_tr_val:.3f} / {top1_va_val:.3f}; "
            f"NDCG@3 train/val: {ndcg_tr:.3f} / {ndcg_va:.3f}."
        )
    except Exception as e:
        lines.append(f"- IRL-B metrics unavailable ({e}).")

lines.append(f"- GRPO (critic-free) trained for {EPOCHS_GRPO} epochs; curve saved.")

if (DATA_DIR/"ope_snips.csv").exists():
    lines.append("- OPE SNIPS computed and saved.")

if (DATA_DIR/"today_offer_recos.csv").exists():
    lines.append("- Demo exported: today_offer_recos.csv (with explanations and mix plot).")

(DATA_DIR/"exec_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n=== Executive Summary ===")
print("\n".join(lines))
print("\n✓ Done. written to: C:/backupcgi/final_bak")
