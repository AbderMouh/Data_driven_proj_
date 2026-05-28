import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score



# This script deos the same thing as the no loop script it s just that here we can loop through multiple lags instead of adding it manually (thanks gemini)
# So here we show that after a certain number of linear combination of lags the RMSE value doesnt change anymore.
# after 4 lag it doesnt change much for the validation set. we also see that more lags makes the model more robust too noise (idk why)
# infact even with a very noisy set after 15-20 lags we can retrieve really well the initial curve (the one without noise)


df = pd.read_csv('./Toy_data/numerical_res.csv')

t = df['time'].values
u0 = df['u1'].values 
y0 = df['y1'].values
y_no_noise = df['y_no_noise'].values
# Define the maximum number of lags to generate (Set to 6)
max_lags = 25

# Lists to hold your lagged arrays dynamically
y_lags = []
u_lags = []

current_y = y0
current_u = u0

for i in range(max_lags):
    current_y = np.hstack([current_y[0], current_y])[:-1]
    y_lags.append(current_y)
    
    current_u = np.hstack([current_u[0], current_u])[:-1]
    u_lags.append(current_u)

# Half-splits for training and validation
y0_red = len(y0) // 2
Y_fit = y0[0:y0_red]
Y_val = y0[y0_red:]

# Slice validation inputs for recursive simulation loops
u_val_matrix = np.column_stack([u[y0_red:] for u in u_lags])

# Text box formatting for plot benchmarks
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

rmse =[]
# --- Loop over 1, 2, 3, 4, 5, and 6 Lags Dynamically ---
for lags in range(1, max_lags + 1):
    
    # 1. Build Feature Matrix (phi) for Fit Phase
    y_cols = [y[:y0_red] for y in y_lags[:lags]]
    u_cols = [u[:y0_red] for u in u_lags[:lags]]
    
    # Generate the square root transformation ONLY for the first lag of y (y1)
    y1_fit = y_cols[0]
    u1_fit = u_cols[0]
    y1_sqrt_col = [ abs(y1_fit)**(1/2)]
    
    # Combine original features and ONLY the single square root feature of y1
    phi_fit = np.column_stack(y_cols + u_cols + y1_sqrt_col)
    
    # 2. Train Model
    model = LinearRegression(fit_intercept=True).fit(phi_fit, Y_fit)
    
    # 3. Fit Phase Prediction & Evaluation
    preds_fit = model.predict(phi_fit)
    rmse_fit = np.sqrt(mean_squared_error(Y_fit, preds_fit))
    r2_fit = r2_score(Y_fit, preds_fit)
    
    # 4. Validation Phase Recursive Simulation Loop
    preds_val = np.zeros(len(Y_val))
    preds_val[0:lags] = Y_val[0:lags] # Seed the loop with real initial history
    
    for k in range(lags, len(Y_val)):
        # Extract past outputs (from our own predictions)
        y_features = [preds_val[k - i] for i in range(1, lags + 1)]
        # Extract past inputs (from validation matrix)
        u_features = [u_val_matrix[k, i] for i in range(lags)]
        
        # Apply the square root ONLY to the first element of y_features, which is preds_val[k-1]
        y1_val_current = y_features[0]
        y1_sqrt_feature = [np.sign(y1_val_current) * np.sqrt(np.abs(y1_val_current))]
        
        # Structure the row vector to match phi_fit exactly: [all_y_linear, all_u_linear, sqrt(y1)]
        Xk = np.array([y_features + u_features + y1_sqrt_feature])
        preds_val[k] = model.predict(Xk)[0]
        
    # 5. Evaluation metrics accumulation
    rmse_val = np.sqrt(mean_squared_error(Y_val, preds_val))
    rmse.append(rmse_val)
    r2_val = r2_score(Y_val, preds_val)

    if  lags == 1 or lags == 3 or lags == 20 :
    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
        t_fit = np.arange(len(Y_fit))
        t_val = np.arange(len(Y_val))
        
        # Plot Fit Phase
        ax1.plot(t_fit, Y_fit, label='True Output (Fit)', color='black', alpha=0.7)
        ax1.plot(t_fit, preds_fit, label=f'Model ({lags} Lag) Prediction', color='crimson', linestyle='--')
        ax1.plot(t_fit, y_no_noise[:len(y_no_noise)//2], label='output without noise', color='green', alpha=0.7)
        ax1.set_title(f'Model with {lags} Lag(s): Fitting Phase')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='upper right')
        text_fit = f'Fit RMSE: {rmse_fit:.4f}\nFit R²: {r2_fit:.4f}'
        ax1.text(0.02, 0.05, text_fit, transform=ax1.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)
        
        # Plot Validation Phase
        ax2.plot(t_val, Y_val, label='True Output (Val)', color='black', alpha=0.7)
        ax2.plot(t_val, preds_val, label=f'Model ({lags} Lag) Simulation', color='darkblue', linestyle='--')
        ax2.plot(t_fit, y_no_noise[len(y_no_noise)//2:], label='output without noise', color='green', alpha=0.7)
        ax2.set_title(f'Model with {lags} Lag(s): Validation Phase (Free-Run Simulation)')
        ax2.set_xlabel('Time Step (k)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(loc='upper right')
        text_val = f'Val RMSE: {rmse_val:.4f}\nVal R²: {r2_val:.4f}'
        ax2.text(0.02, 0.05, text_val, transform=ax2.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)
        
        plt.tight_layout()

plt.show()