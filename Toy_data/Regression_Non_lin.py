import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score



df = pd.read_csv('./Toy_data/numerical_res.csv')

t = df['time'].values
u0 = df['u1'].values 
y0 = df['y1'].values
y_no_noise = df['y_no_noise'].values


# Define the maximum number of lags to generate 
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

rmse =[]
# --- Loop over 1, 2, 3, 4, 5, and 6 Lags Dynamically ---
for lags in range(1, max_lags + 1):
    
    # 1. Build Feature Matrix (phi) for Fit Phase
    y_cols = [y[:y0_red] for y in y_lags[:lags]]
    u_cols = [u[:y0_red] for u in u_lags[:lags]]
    
    # Generate the square root transformation ONLY for the first lag of y (y1)
    y1_fit = y_cols[0]
    u1_fit = u_cols[0]
    y1_sqrt_col = [abs(y1_fit)**(1/2)]
    
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
    

    for k in range(0, len(Y_val)):
    
        # Extract past outputs: if k - i < 0, it means it's before the start, so use Y_val[0]
        y_features = [preds_val[k - i] if (k - i) >= 0 else Y_val[0] for i in range(1, lags + 1)]
        
        # Extract past inputs: do the same zero-padding if the index is out of bounds
        u_features = [u_val_matrix[k, i] for i in range(lags)] 

        y1_val_current = y_features[0]
        y1_sqrt_feature = [np.sign(y1_val_current) * np.sqrt(np.abs(y1_val_current))]

        # Combine into your final row vector [y_linear, u_linear, sqrt(y1)]
        Xk = np.array([y_features + u_features +y1_sqrt_feature ])
        
        # Predict step k
        preds_val[k] = model.predict(Xk)[0]


    # 5. Evaluation metrics accumulation
    rmse_val = np.sqrt(mean_squared_error(Y_val, preds_val))
    rmse.append(rmse_val)
    r2_val = r2_score(Y_val, preds_val)




    if  lags == 1 or lags == 10 or lags == 20  :
    
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