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

# Half-splits for training and validation
y0_red = len(y0) // 2
Y_fit = y0[0:y0_red]
Y_val = y0[y0_red:]

# So we shift by one add the initial value as the first value and then remove the las value to keep the same dimensions
for i in range(1,max_lags+1):

    y_lags.append(np.hstack([np.ones(i)*y0[0], y0[:-i]]))

    u_lags.append(np.hstack([np.ones(i)*u0[0], u0[:-i]]))



# Slice validation inputs for recursive simulation loops
u_val_matrix = np.column_stack([u[y0_red:] for u in u_lags])

# Text box formatting for plot benchmarks
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

rmse =[] #list to store the root mean square error to plot it later


# --- Loop over 1, 2, 3, ... lags
for lags in range(1, max_lags + 1):
    
    # 1. Build Feature Matrix (phi) for Fit Phase
    y_cols = [y[:y0_red] for y in y_lags[:lags]]
    u_cols = [u[:y0_red] for u in u_lags[:lags]]
    phi_fit = np.column_stack(y_cols + u_cols)
    
    
    # 2. Train Model
    model = LinearRegression(fit_intercept=True).fit(phi_fit, Y_fit)
    
    # 3. Fit Phase Prediction & Evaluation
    preds_fit = model.predict(phi_fit)
    rmse_fit = np.sqrt(mean_squared_error(Y_fit, preds_fit))
    r2_fit = r2_score(Y_fit, preds_fit)
    
    # 4. Validation Phase Recursive Simulation Loop
    # list that will contain all of our predicted y values
    # since for each step we need all the past input we have to compute it incrementally 

    preds_val = np.zeros(len(Y_val))
    
    
    for k in range(0, len(Y_val)):
    
        # Extract past outputs: if k - i < 0, it means it's before the start, so use Y_val[0]
        y_features = [preds_val[k - i] if (k - i) >= 0 else Y_val[0] for i in range(1, lags + 1)]
        
        # Extract past inputs: do the same zero-padding if the index is out of bounds
        u_features = [u_val_matrix[k, i] for i in range(lags)] 

        # Combine into your final row vector [y_linear, u_linear, sqrt(y1)]
        Xk = np.array([y_features + u_features ])
        
        # Predict step k
        preds_val[k] = model.predict(Xk)[0]
        

        
    rmse_val = np.sqrt(mean_squared_error(Y_val[lags:], preds_val[lags:]))
    rmse.append(rmse_val)
    r2_val = r2_score(Y_val[lags:], preds_val[lags:])
    






    if  lags == 1 or lags == 3 or lags == 20 or lags == 90:
    
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

fig, ax3 = plt.subplots(figsize=(12, 8))

# Plot the RMSE array against the number of lags

ax3.plot(np.arange(1, len(rmse) + 1), rmse, label='Root Mean Square Error', color='blue', marker='.')

ax3.set_title('Validation RMSE depending on the Number of Lags')
ax3.set_xlabel('Number of Lags')
ax3.set_ylabel('RMSE Value')
ax3.grid(True)
ax3.legend()
plt.show()