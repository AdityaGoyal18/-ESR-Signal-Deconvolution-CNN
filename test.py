import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

from models.esr_model import ESRModel

# load dataset
X = np.load("data/X.npy")
Y = np.load("data/Y.npy")

# load model
model = ESRModel()
model.load_state_dict(torch.load("esr_model.pth"))
model.eval()

# take one sample
x = torch.tensor(X[10]).unsqueeze(0).float()

# prediction
pred = model(x).detach().numpy()[0]

true = Y[10]
composite = X[10]

# reconstructed composite
reconstructed = pred[0] + pred[1]

# error
error = true - pred

from sklearn.metrics import mean_squared_error

# ---------- Performance Metrics ----------

# MSE
mse1 = mean_squared_error(true[0], pred[0])
mse2 = mean_squared_error(true[1], pred[1])

# RMSE
rmse1 = np.sqrt(mse1)
rmse2 = np.sqrt(mse2)

# MAE
mae1 = mean_absolute_error(true[0], pred[0])
mae2 = mean_absolute_error(true[1], pred[1])

# R2 Score
r2_1 = r2_score(true[0], pred[0])
r2_2 = r2_score(true[1], pred[1])

# Pearson correlation
corr1, _ = pearsonr(true[0], pred[0])
corr2, _ = pearsonr(true[1], pred[1])

# Peak error
peak_error1 = abs(np.max(true[0]) - np.max(pred[0]))
peak_error2 = abs(np.max(true[1]) - np.max(pred[1]))

# SNR
snr1 = 10*np.log10(np.sum(true[0]**2)/np.sum((true[0]-pred[0])**2))
snr2 = 10*np.log10(np.sum(true[1]**2)/np.sum((true[1]-pred[1])**2))

# prediction
pred = model(x).detach().numpy()[0]

true = Y[10]
composite = X[10]


print("\n----- Model Performance -----")

print("\nRadical 1 Metrics")
print("MSE:", mse1)
print("RMSE:", rmse1)
print("MAE:", mae1)
print("R2 Score:", r2_1)
print("Correlation:", corr1)
print("Peak Error:", peak_error1)
print("SNR:", snr1)

print("\nRadical 2 Metrics")
print("MSE:", mse2)
print("RMSE:", rmse2)
print("MAE:", mae2)
print("R2 Score:", r2_2)
print("Correlation:", corr2)
print("Peak Error:", peak_error2)
print("SNR:", snr2)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------- Confusion Matrix (Peak Detection) ----------

threshold = 0.2   # threshold to decide peak presence

# convert signals to binary peak detection
true_peaks = (true[0] > threshold).astype(int)
pred_peaks = (pred[0] > threshold).astype(int)

# compute confusion matrix
cm = confusion_matrix(true_peaks, pred_peaks)

print("\nConfusion Matrix (Radical 1 Peak Detection):")
print(cm)

plt.figure(figsize=(5,4))

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["No Peak","Peak"])

disp.plot(cmap="Blues", values_format="d")

plt.title("Confusion Matrix for Radical 1 Peak Detection")
plt.show()


true_peaks2 = (true[1] > threshold).astype(int)
pred_peaks2 = (pred[1] > threshold).astype(int)

cm2 = confusion_matrix(true_peaks2, pred_peaks2)

plt.figure(figsize=(5,4))

disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,
                               display_labels=["No Peak","Peak"])

disp2.plot(cmap="Greens", values_format="d")

plt.title("Confusion Matrix for Radical 2 Peak Detection")
plt.show()

# Create figure
plt.figure(figsize=(16,10))

# Plot 1: All signals
plt.subplot(2,3,1)
plt.plot(composite,label="Composite ESR")
plt.plot(true[0],label="True Radical 1")
plt.plot(true[1],label="True Radical 2")
plt.plot(pred[0],'--',label="Pred Radical 1")
plt.plot(pred[1],'--',label="Pred Radical 2")
plt.title("All Signals")
plt.legend()

# Plot 2: Composite vs true radicals
plt.subplot(2,3,2)
plt.plot(composite,label="Composite ESR")
plt.plot(true[0],'--',label="True Radical 1")
plt.plot(true[1],'--',label="True Radical 2")
plt.title("Composite vs True Components")
plt.legend()

# Plot 3: Radical 1 comparison
plt.subplot(2,3,3)
plt.plot(true[0],label="True Radical 1")
plt.plot(pred[0],'--',label="Pred Radical 1")
plt.title("Radical 1: True vs Predicted")
plt.legend()

# Plot 4: Radical 2 comparison
plt.subplot(2,3,4)
plt.plot(true[1],label="True Radical 2")
plt.plot(pred[1],'--',label="Pred Radical 2")
plt.title("Radical 2: True vs Predicted")
plt.legend()

# Plot 5: Composite vs Reconstructed
plt.subplot(2,3,5)
plt.plot(composite,label="Composite")
plt.plot(reconstructed,'--',label="Reconstructed (Pred1 + Pred2)")
plt.title("Composite vs Reconstructed")
plt.legend()

# Plot 6: Error Plot
plt.subplot(2,3,6)
plt.plot(error[0],label="Radical1 Error")
plt.plot(error[1],label="Radical2 Error")
plt.title("Prediction Error")
plt.legend()

plt.tight_layout()

plt.show()

# ================= ADDITIONAL ANALYSIS =================

# ---------- RMSE FUNCTION ----------
def compute_rmse(true, pred):
    return np.sqrt(np.mean((true - pred)**2))


# ---------- OVERLAP FUNCTION ----------
def compute_overlap(r1, r2):
    overlap = np.sum(np.minimum(np.abs(r1), np.abs(r2)))
    total = np.sum(np.maximum(np.abs(r1), np.abs(r2))) + 1e-8
    return overlap / total


# ---------- PARITY PLOT ----------
true_flat = true.flatten()
pred_flat = pred.flatten()

plt.figure(figsize=(6,6))
plt.scatter(true_flat, pred_flat, alpha=0.3)

plt.plot(
    [true_flat.min(), true_flat.max()],
    [true_flat.min(), true_flat.max()],
    'r--'
)

plt.xlabel("True Intensity")
plt.ylabel("Predicted Intensity")
plt.title("Parity Plot (True vs Predicted)")

plt.show()


# ---------- ERROR DISTRIBUTION ----------
error_flat = (true - pred).flatten()

plt.figure(figsize=(6,4))
plt.hist(error_flat, bins=50)

plt.xlabel("Error")
plt.ylabel("Frequency")
plt.title("Error Distribution")

plt.show()


# ---------- ERROR vs OVERLAP ----------
errors = []
overlap_vals = []

for i in range(len(X)):

    x_sample = torch.tensor(X[i]).unsqueeze(0).float()
    pred_sample = model(x_sample).detach().numpy()[0]
    true_sample = Y[i]

    rmse = compute_rmse(true_sample[0], pred_sample[0])
    errors.append(rmse)

    overlap = compute_overlap(true_sample[0], true_sample[1])
    overlap_vals.append(overlap)

errors = np.array(errors)
overlap_vals = np.array(overlap_vals)

plt.figure(figsize=(6,5))
plt.scatter(overlap_vals, errors, alpha=0.5)

plt.xlabel("Overlap Ratio")
plt.ylabel("RMSE")
plt.title("Error vs Overlap")

plt.show()


# ---------- BASELINE METHOD ----------
from scipy.signal import find_peaks

def baseline_peak_split(x):

    mid = len(x)//2

    y1 = np.copy(x)
    y2 = np.copy(x)

    y1[mid:] = 0
    y2[:mid] = 0

    return y1, y2


# ---------- BASELINE vs CNN COMPARISON ----------
baseline_errors = []
cnn_errors = []

for i in range(len(X)):

    x_sample = X[i]
    true_sample = Y[i]

    # baseline
    y1_base, _ = baseline_peak_split(x_sample)

    # CNN prediction
    x_tensor = torch.tensor(x_sample).unsqueeze(0).float()
    pred_sample = model(x_tensor).detach().numpy()[0]

    baseline_rmse = compute_rmse(true_sample[0], y1_base)
    cnn_rmse = compute_rmse(true_sample[0], pred_sample[0])

    baseline_errors.append(baseline_rmse)
    cnn_errors.append(cnn_rmse)

print("\n===== BASELINE vs CNN =====")
print("Baseline RMSE:", np.mean(baseline_errors))
print("CNN RMSE:", np.mean(cnn_errors))

# ---------- BAR PLOT: BASELINE vs CNN ----------

plt.figure(figsize=(6,4))

labels = ["Baseline", "CNN"]
values = [np.mean(baseline_errors), np.mean(cnn_errors)]

plt.bar(labels, values)

plt.ylabel("RMSE")
plt.title("Baseline vs CNN Performance")

for i, v in enumerate(values):
    plt.text(i, v + 0.001, f"{v:.4f}", ha='center')

plt.show()

# ---------- DISTRIBUTION PLOT ----------

plt.figure(figsize=(6,4))

plt.hist(baseline_errors, bins=50, alpha=0.5, label="Baseline")
plt.hist(cnn_errors, bins=50, alpha=0.5, label="CNN")

plt.xlabel("RMSE")
plt.ylabel("Frequency")
plt.title("Error Distribution: Baseline vs CNN")

plt.legend()

plt.show()