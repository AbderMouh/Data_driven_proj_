import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# =====================================================================
# 1. Load CSV and Create Interpolation Function for q_in(t)
# =====================================================================



a = "sine"

if a == "sine":

    S = 0.020 # surface de rayon 8 cm +-
    c = 0.02 # Valve/discharge constant
    h0 = 0.08 # Initial tank height at t = 0


    df = pd.read_csv('./Real_data/dataBenchmark.csv') # je vole l'input du real dataset >:D
    q_data = df['uEst'].values * 0.002  # Pour avoir des valeur realiste en m3/s vu que de base c est en volt
    dt = 4/1024    
                                        # Adjust this to your actual time increment
    t_data = np.arange(len(q_data)) * dt                 

if a == "step":

    S = 0.020 # surface de rayon 8 cm +-
    c = 0.045# Valve/discharge constant
    h0 = 0.08  # Initial tank height at t = 0

    # Your time configuration
    dt = 4/ 1024  

    # Define the custom sequence of steps you want (similar to the image)
    # You can change these numbers to match whatever levels you want to test
    step_levels = [0.004, 0.015, 0.001,0.010,0.006]

    # Define how long each plateau should last (in seconds)
    seconds_per_step = 5

    # Calculate how many data points are needed to fill that time duration
    samples_per_step = int(seconds_per_step / dt)

    # Generate q_data by repeating each level for the calculated number of samples
    q_data = np.repeat(step_levels, samples_per_step)

    # Generate the corresponding time array matching the length of q_data
    t_data = np.arange(len(q_data)) * dt

    



# # Create a continuous lookup function from the discrete CSV data
# # 'bounds_error=False, fill_value="extrapolate"' prevents crashing if the solver looks slightly past your data
q_in_lookup = interp1d(t_data, q_data, kind='linear', bounds_error=False, fill_value="extrapolate")

  






# =====================================================================
# 2. Define System Parameters and Differential Equation
# =====================================================================



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

t_eval = np.linspace(t_data[0], t_data[-1], len(t_data)) 

solution = solve_ivp(tank_system, t_span, [h0], t_eval=t_data)



noise = np.random.normal(loc=0, scale=0.002, size=solution.y[0].shape)


noisy = solution.y[0] + noise






# 1. Gather the solved time values and the corresponding calculated heights
# solution.t contains the timestamps; solution.y[0] contains the calculated h(t) values
results_data = {

    'time': solution.t,
    'u1': q_in_lookup(solution.t),  
    'y1': noisy,
    'y_no_noise':solution.y[0]
    

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
ax2.plot(solution.t, noisy, color='blue', linewidth=2, label='Simulated Height $h(t)$')
ax2.set_xlabel('Time ($t$)')
ax2.set_ylabel('Height ($h$)')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
