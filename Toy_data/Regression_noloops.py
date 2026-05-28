

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('./Toy_data/numerical_res.csv')


t = df['time'].values
u0= df['u1'].values #uk
y0 = df['y1'].values


# Define how many lags you want to create
max_lags = 10

# Lists to hold your lagged arrays dynamically
y_lags = []
u_lags = []

# Temporary trackers starting with the original signals
current_y = y0
current_u = u0

for i in range(max_lags):
    # Shift Y: repeat the first element at the front, drop the last element
    current_y = np.hstack([current_y[0], current_y])[:-1]
    y_lags.append(current_y)
    
    # Shift U: repeat the first element at the front, drop the last element
    current_u = np.hstack([current_u[0], current_u])[:-1]
    u_lags.append(current_u)



# the idea is to find yk based on past inputs and outputs
# for a first order Differential equation dy/dt = f(u,y)
# The discretized veriosn of it can be written as yk = yk-1 + delta_t*f(uk-1,yk-1)
# we see that to describe the behavior of this system we only need one step past information, this is called 1 lag
# Now what we will do is check if more lag could improve the model's prediciton
# to make it easy we will just use a linear combination of lags

# y = a1*y1+a2*y2+b1*u1+b2*u2
# y = phi*w



y1 = y_lags[0]
u1 = u_lags[0]
y2 = y_lags[1]
u2 = u_lags[1]

y1_red = len(y1)//2
u1_red = len(u1)//2
y2_red = len(y2)//2
u2_red = len(u2)//2
y0_red = len(u0)//2

print(y1_red)

#1 lag linear ARX model
y1_fit = y1[0:y1_red]
y1_val = y1[y1_red:]

y2_fit = y2[0:y2_red]
y2_val = y2[y2_red:]

u1_fit = u1[:u1_red]
u1_val = u1[u1_red:]

u2_fit = u2[:u2_red]
u2_val = u2[u2_red:]

phi1_fit = np.column_stack((y1_fit, u1_fit**(1/2),u1_fit))
#phi1_val = np.column_stack((y1_val, u1_val))

phi2_fit = np.column_stack((y1_fit,y2_fit, u1_fit**(1/2),u1_fit, u2_fit))
#phi2_val = np.column_stack((y1_val, y2_val, u1_val, u2_val))


Y_fit = y0[0:y0_red]
Y_val = y0[y0_red:]


print(np.shape(Y_fit))
print(np.shape(phi1_fit))


ols_model1 = LinearRegression(fit_intercept=True)
ols_model1.fit(phi1_fit, Y_fit)

ols_model2 = LinearRegression(fit_intercept=True)
ols_model2.fit(phi2_fit, Y_fit)


ols_preds1_fit = ols_model1.predict(phi1_fit)
ols_rmse1_fit = np.sqrt(mean_squared_error(Y_fit, ols_preds1_fit))
ols_r21_fit = r2_score(Y_fit, ols_preds1_fit)


ols_preds1_val = np.zeros(len(Y_val))

# initial condition
ols_preds1_val[0] = Y_val[0]

for k in range(1, len(Y_val)):

    Xk = np.array([[ols_preds1_val[k-1],u1_val[k]**(1/2),u1_val[k]]])

    ols_preds1_val[k] = ols_model1.predict(Xk)[0]




ols_rmse1_val = np.sqrt(mean_squared_error(Y_val, ols_preds1_val))
ols_r21_val = r2_score(Y_val, ols_preds1_val)



ols_preds2_fit = ols_model2.predict(phi2_fit)
ols_rmse2_fit = np.sqrt(mean_squared_error(Y_fit, ols_preds2_fit))
ols_r22_fit = r2_score(Y_fit, ols_preds2_fit)


ols_preds2_val = np.zeros(len(Y_val))

# initialize first two samples
ols_preds2_val[0] = Y_val[0]
ols_preds2_val[1] = Y_val[1]

for k in range(2, len(Y_val)):

    Xk = np.array([[
        ols_preds2_val[k-1],
        ols_preds2_val[k-2],
        u1_val[k]**(1/2),u1_val[k],
        u2_val[k]
    ]])

    ols_preds2_val[k] = ols_model2.predict(Xk)[0]


ols_rmse2_val = np.sqrt(mean_squared_error(Y_val, ols_preds2_val))
ols_r22_val = r2_score(Y_val, ols_preds2_val)










t = np.arange(len(Y_fit))

# Custom box style for the benchmark text
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# --- Figure 1: Model 1 (1 Lag ARX) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharey=True)

# Fit Phase
t_fit = np.arange(len(Y_fit))
ax1.plot(t_fit, Y_fit, label='True Output (Fit)', color='black', alpha=0.7)
ax1.plot(t_fit, ols_preds1_fit, label='Model 1 Prediction (Fit)', color='crimson', linestyle='--')
ax1.set_title('Model 1 (1 Lag): Fitting Phase')
ax1.set_ylabel('Amplitude')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right')

# Add Fit Benchmarks
text_m1_fit = f'Fit RMSE: {ols_rmse1_fit:.4f}\nFit R²: {ols_r21_fit:.4f}'
ax1.text(0.02, 0.05, text_m1_fit, transform=ax1.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)

# Validation Phase
t_val = np.arange(len(Y_val))
ax2.plot(t_val, Y_val, label='True Output (Val)', color='black', alpha=0.7)
ax2.plot(t_val, ols_preds1_val, label='Model 1 Prediction (Val)', color='darkred', linestyle='--')
ax2.set_title('Model 1 (1 Lag): Validation Phase')
ax2.set_xlabel('Time Step (k)')
ax2.set_ylabel('Amplitude')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

# Add Validation Benchmarks (Using variables recalculated from previous script)
text_m1_val = f'Val RMSE: {ols_rmse1_val:.4f}\nVal R²: {ols_r21_val:.4f}'
ax2.text(0.02, 0.05, text_m1_val, transform=ax2.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)

plt.tight_layout()


# --- Figure 2: Model 2 (2 Lags ARX) ---
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 8), sharey=True)

# Fit Phase
ax3.plot(t_fit, Y_fit, label='True Output (Fit)', color='black', alpha=0.7)
ax3.plot(t_fit, ols_preds2_fit, label='Model 2 Prediction (Fit)', color='dodgerblue', linestyle=':')
ax3.set_title('Model 2 (2 Lags): Fitting Phase')
ax3.set_ylabel('Amplitude')
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='upper right')

# Add Fit Benchmarks
text_m2_fit = f'Fit RMSE: {ols_rmse2_fit:.4f}\nFit R²: {ols_r22_fit:.4f}'
ax3.text(0.02, 0.05, text_m2_fit, transform=ax3.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)

# Validation Phase
ax4.plot(t_val, Y_val, label='True Output (Val)', color='black', alpha=0.7)
ax4.plot(t_val, ols_preds2_val, label='Model 2 Prediction (Val)', color='navy', linestyle=':')
ax4.set_title('Model 2 (2 Lags): Validation Phase')
ax4.set_xlabel('Time Step (k)')
ax4.set_ylabel('Amplitude')
ax4.grid(True, linestyle=':', alpha=0.6)
ax4.legend(loc='upper right')

# Add Validation Benchmarks
text_m2_val = f'Val RMSE: {ols_rmse2_val:.4f}\nVal R²: {ols_r22_val:.4f}'
ax4.text(0.02, 0.05, text_m2_val, transform=ax4.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)

plt.tight_layout()
plt.show()

