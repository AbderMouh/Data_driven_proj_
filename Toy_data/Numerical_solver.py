import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# =====================================================================
# 1. Load CSV and Create Interpolation Function for q_in(t)
# =====================================================================

# --- SCENARIO A: Your CSV has two columns: [time, q_in] ---
df = pd.read_csv('./Real_data/dataBenchmark.csv') # je vole l'input du real dataset >:D
q_data = df['uEst'].values * 0.001  # Pour avoir des valeur realiste en m3/s vu que de base c est en volt
dt = 4/1024                                            # Adjust this to your actual time increment
t_data = np.arange(len(q_data)) * dt                 # Creates time array from 0 to 102.3 seconds






# Create a continuous lookup function from the discrete CSV data
# 'bounds_error=False, fill_value="extrapolate"' prevents crashing if the solver looks slightly past your data
q_in_lookup = interp1d(t_data, q_data, kind='linear', bounds_error=False, fill_value="extrapolate")


# =====================================================================
# 2. Define System Parameters and Differential Equation
# =====================================================================
S = 0.005 # surface de rayon 4 cm +-
c = 0.012 # Valve/discharge constant
h0 = 0.1   # Initial tank height at t = 0

def tank_system(t, h):
    h_val = max(h[0], 0.0)  # Safeguard against negative heights due to numerical overshoot

    # Get the value of q_in at the exact time 't' from the CSV lookup
    current_q_in = q_in_lookup(t)

    # dh/dt = (q_in - q_out) / S
    dh_dt = (current_q_in - c * np.sqrt(h_val)) / S
    return [dh_dt]


# =====================================================================
# 3. Solve the ODE
# =====================================================================
# Define simulation span based exactly on the time limits of your CSV file
t_span = (t_data[0], t_data[-1])

t_eval = np.linspace(t_data[0], t_data[-1], 1000) # 1000 points for a smooth plot

solution = solve_ivp(tank_system, t_span, [h0], t_eval=t_eval)





# 1. Gather the solved time values and the corresponding calculated heights
# solution.t contains the timestamps; solution.y[0] contains the calculated h(t) values
results_data = {
    'time': solution.t,
    'u1': q_in_lookup(solution.t),  
    'y1': solution.y[0]

}



# 3. Convert the dictionary into a Pandas DataFrame
df_results = pd.DataFrame(results_data)

# 4. Export to a new CSV file
# 'index=False' prevents Pandas from adding an unnamed column of row numbers (0, 1, 2...)
df_results.to_csv('./Toy_data/numerical_res.csv', index=False)




# =====================================================================
# 4. Plot the Results side-by-side
# =====================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: Input Flow rate from your CSV
ax1.plot(t_data, q_data, color='orange', label='Input Flow $q_{in}(t)$ (from CSV)')
ax1.set_ylabel('Flow Rate ($q_{in}$)')
ax1.grid(True)
ax1.legend()

# Plot 2: Simulated Tank Height
ax2.plot(solution.t, solution.y[0], color='blue', linewidth=2, label='Simulated Height $h(t)$')
ax2.set_xlabel('Time ($t$)')
ax2.set_ylabel('Height ($h$)')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
