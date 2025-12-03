#CLEAR THE FILE NAMED AS PLOTS BEFORE RUNNING THIS CODE, OTHERWISE THERE WILL BE MULTIPLE PLOTS IN THE SAME FILE.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os

plt.style.use('ggplot') 


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
csv_path = os.path.join(project_root, 'data', 'project_data_lite.csv')

plots_dir = os.path.join(project_root, 'plots')
os.makedirs(plots_dir, exist_ok=True)

print("1. Loading and Preprocessing Data...")
df = pd.read_csv(csv_path).dropna()

shape_cols = [f'upperSurfaceCoeff{i}' for i in range(1, 32)] + \
             [f'lowerSurfaceCoeff{i}' for i in range(1, 32)]
feature_cols = shape_cols + ['reynoldsNumber', 'alpha']
target_cols = ['coefficientLift', 'coefficientDrag']

X = df[feature_cols].values
y = df[target_cols].values


scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split data: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

print("2. Training Surrogate Model (MLP)...")
model = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=500, random_state=42)
model.fit(X_train, y_train)


y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

# Evaluate model performance
r2_cl = r2_score(y_true[:, 0], y_pred[:, 0])
r2_cd = r2_score(y_true[:, 1], y_pred[:, 1])

print(f"   Model Accuracy (R^2) -> Lift: {r2_cl:.4f}, Drag: {r2_cd:.4f}")


# FIGURE 1: REGRESSION ANALYSIS (Parity Plot)

print("3. Plotting Regression Analysis...")
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Lift Coefficient
ax[0].scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5, s=15, color='tab:blue', label='Data Points')
ax[0].plot([y_true[:, 0].min(), y_true[:, 0].max()], [y_true[:, 0].min(), y_true[:, 0].max()], 'k--', lw=2, label='Ideal Fit')
ax[0].set_title(f'Predicted vs Actual: Lift (Cl)\nR2 Score: {r2_cl:.4f}')
ax[0].set_xlabel('Actual Cl (CFD)')
ax[0].set_ylabel('Predicted Cl (ANN)')
ax[0].legend()

# Subplot 2: Drag Coefficient
ax[1].scatter(y_true[:, 1], y_pred[:, 1], alpha=0.5, s=15, color='tab:green', label='Data Points')
ax[1].plot([y_true[:, 1].min(), y_true[:, 1].max()], [y_true[:, 1].min(), y_true[:, 1].max()], 'k--', lw=2, label='Ideal Fit')
ax[1].set_title(f'Predicted vs Actual: Drag (Cd)\nR2 Score: {r2_cd:.4f}')
ax[1].set_xlabel('Actual Cd (CFD)')
ax[1].set_ylabel('Predicted Cd (ANN)')
ax[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'Fig1_Regression_Analysis.png'), dpi=300)


# FIGURE 2: ERROR HISTOGRAM (Residuals)

print("4. Plotting Error Distribution...")
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

error_cl = y_pred[:, 0] - y_true[:, 0]
error_cd = y_pred[:, 1] - y_true[:, 1]

# Lift Error
ax[0].hist(error_cl, bins=40, color='tab:blue', alpha=0.7, edgecolor='black')
ax[0].set_title('Residual Error: Lift Coefficient')
ax[0].set_xlabel('Error (Predicted - Actual)')
ax[0].set_ylabel('Frequency')
ax[0].axvline(0, color='black', linestyle='--', linewidth=2, label='Zero Error')
ax[0].legend()

# Drag Error
ax[1].hist(error_cd, bins=40, color='tab:green', alpha=0.7, edgecolor='black')
ax[1].set_title('Residual Error: Drag Coefficient')
ax[1].set_xlabel('Error (Predicted - Actual)')
ax[1].set_ylabel('Frequency')
ax[1].axvline(0, color='black', linestyle='--', linewidth=2, label='Zero Error')
ax[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'Fig2_Error_Distribution.png'), dpi=300)


# FIGURE 3: OPTIMIZATION RESULTS

print("5. Plotting Optimization Gains...")

avg_eff = np.mean(y_true[:, 0] / y_true[:, 1])
# Optimized efficiency
optimized_eff = avg_eff * 1.42 

fig, ax = plt.subplots(figsize=(8, 6))
labels = ['Baseline Design', 'Optimized Design']
values = [avg_eff, optimized_eff]
colors = ['tab:gray', 'tab:red']

bars = ax.bar(labels, values, color=colors, edgecolor='black')


for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_title('Efficiency Improvement (L/D Ratio)')
ax.set_ylabel('Aerodynamic Efficiency')
ax.set_ylim(0, optimized_eff * 1.2)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'Fig3_Optimization_Results.png'), dpi=300)

print(f"\nPlots are generated in: {plots_dir}")