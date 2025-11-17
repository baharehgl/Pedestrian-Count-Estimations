"""
VAE-based data augmentation for pedestrian count prediction (pm_tot)

Pipeline:
  1) Load preprocessed numeric data: processed/X.csv, processed/y.csv
  2) Train/test split on REAL data (e.g., 80/20)
  3) Baseline model: RandomForest on real train -> evaluate on real test
  4) Train a small VAE on scaled X_train
  5) Sample synthetic feature vectors X_synth from VAE
  6) Label X_synth using baseline RF (pseudo-labels)
  7) Train a new RF on (X_train + X_synth, y_train + y_synth_pseudo)
  8) Evaluate augmented RF on the same real test set

You can compare:
  - Baseline RF vs RF + VAE-augmented data

NOTE:
  - VAE learns only p(X); we use RF as "teacher" to give labels.
  - This is experimental (data is tiny ~100 rows), but illustrates how generative AI fits in.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim

# --------------------- CONFIG ------------------------
PROCESSED_DIR = "./processed"          # folder from your previous preprocessing step
X_PATH = os.path.join(PROCESSED_DIR, "X.csv")
Y_PATH = os.path.join(PROCESSED_DIR, "y.csv")
TARGET_COL = "pm_tot"                  # name used in preprocessing
TEST_SIZE = 0.2                        # 20% real test set
RANDOM_STATE = 42

LATENT_DIM = 8                         # small latent space (can tweak)
HIDDEN_DIM = 32                        # VAE hidden layer size
EPOCHS = 400                           # train epochs for VAE
BATCH_SIZE = 16
LR = 1e-3                              # learning rate for VAE
N_SYNTH = 300                          # number of synthetic samples to generate
# ----------------------------------------------------


# =============== 1) Load data ========================
if not (os.path.exists(X_PATH) and os.path.exists(Y_PATH)):
    raise FileNotFoundError("Could not find processed/X.csv or processed/y.csv. Run preprocessing first.")

X = pd.read_csv(X_PATH)
y_df = pd.read_csv(Y_PATH)

if TARGET_COL not in y_df.columns:
    TARGET_COL = y_df.columns[0]  # fallback
y = y_df[TARGET_COL].astype(float).values

print(f"[INFO] Loaded X: {X.shape}, y: {y.shape}")

# ---------------- Train/test split (REAL data only) ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")


# =============== 2) Baseline RF on real data =======================
rf_base = RandomForestRegressor(
    n_estimators=600,
    max_depth=6,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)

def print_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{name}] MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
    return mae, rmse, r2

print_metrics("Baseline RF (real only)", y_test, y_pred_base)


# =============== 3) VAE definition ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

input_dim = X_train.shape[1]

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        x_recon = self.fc_out(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction (MSE) + KL divergence
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # KL: D_KL(q(z|x) || p(z)) with p ~ N(0, I)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld /= x.shape[0]  # average per batch
    return recon_loss + 1e-3 * kld  # small KL weight, can tune


# =============== 4) Train VAE on scaled X_train ====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # not used by VAE, but could be useful

vae = VAE(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(vae.parameters(), lr=LR)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)

n_batches = int(np.ceil(X_train_tensor.shape[0] / BATCH_SIZE))

vae.train()
for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(X_train_tensor.shape[0])
    X_shuffled = X_train_tensor[perm]
    epoch_loss = 0.0

    for i in range(n_batches):
        batch = X_shuffled[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        optimizer.zero_grad()
        recon, mu, logvar = vae(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.size(0)

    epoch_loss /= X_train_tensor.shape[0]
    if epoch % 50 == 0 or epoch == 1:
        print(f"[VAE] Epoch {epoch}/{EPOCHS}, Loss = {epoch_loss:.4f}")

vae.eval()

# Optional: check reconstruction quality roughly
with torch.no_grad():
    recon_all, _, _ = vae(X_train_tensor)
recon_error = nn.functional.mse_loss(recon_all, X_train_tensor).item()
print(f"[VAE] Final reconstruction MSE on train (scaled): {recon_error:.4f}")


# =============== 5) Sample synthetic X from VAE =====================
vae.eval()
n_synth = N_SYNTH

with torch.no_grad():
    # sample z ~ N(0, I)
    z = torch.randn(n_synth, LATENT_DIM).to(device)
    X_synth_scaled = vae.decode(z)
    X_synth_scaled = X_synth_scaled.cpu().numpy()

# inverse scale to original feature space
X_synth = scaler.inverse_transform(X_synth_scaled)

print(f"[INFO] Generated {X_synth.shape[0]} synthetic feature vectors from VAE.")


# =============== 6) Pseudo-label synthetic samples ==================
# Use baseline RF as "teacher"
y_synth_pseudo = rf_base.predict(X_synth)
print("[INFO] Pseudo-labels for synthetic data generated via baseline RF.")


# =============== 7) Train augmented RF ==============================
X_aug = np.vstack([X_train, X_synth])
y_aug = np.concatenate([y_train, y_synth_pseudo])

rf_aug = RandomForestRegressor(
    n_estimators=600,
    max_depth=6,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_aug.fit(X_aug, y_aug)
y_pred_aug = rf_aug.predict(X_test)

print_metrics("Augmented RF (real + VAE synthetic)", y_test, y_pred_aug)


# =============== 8) Simple comparison plot ==========================
labels = ["Baseline RF", "RF + VAE synthetic"]
mae_vals = [
    mean_absolute_error(y_test, y_pred_base),
    mean_absolute_error(y_test, y_pred_aug),
]
rmse_vals = [
    np.sqrt(mean_squared_error(y_test, y_pred_base)),
    np.sqrt(mean_squared_error(y_test, y_pred_aug)),
]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(6,4))
plt.bar(x - width/2, mae_vals, width, label="MAE")
plt.bar(x + width/2, rmse_vals, width, label="RMSE")
plt.xticks(x, labels, rotation=15)
plt.ylabel("Error")
plt.title("Baseline vs VAE-Augmented RF\n(Errors on REAL test set)")
plt.legend()
plt.tight_layout()
plt.show()
