import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 1) Define the SHO system
# Simple Harmonic Oscillator

# x** = -x
# x* = v
# v* = -x

def sho(state, t):
    x = state[0]
    v = state[1]
    dxdt = v
    dvdt = -x
    return [dxdt, dvdt]

def sho_damped(state, t, damping = 0.1):
    x, v = state
    dxdt = v
    dvdt = -x - damping * v
    return [dxdt, dvdt]

# 2) Simulate the system

t = np.linspace(0, 20, 1000)  # time points
y0 = [1.0, 0.0]

#sol_undamped
sol_undamped = odeint(sho, y0, t) #shape (len(t), 2)

#sol_damped
damped_coeff = .1
sol_damped = odeint(sho_damped, y0, t, args=(damped_coeff,))

# 3) Prepare and fit SINDy

libary = ps.PolynomialLibrary(degree=2) #use polynomial libary (degree 2)
optimizer  = ps.STLSQ(threshold=0.05) # tweak threshold if you get extra term

model_undamped = ps.SINDy(feature_library=libary, optimizer=optimizer)
model_damped = ps.SINDy(feature_library=libary, optimizer=optimizer)

model_undamped.fit(sol_undamped, t=t)
model_damped.fit(sol_damped, t=t)

print("undamped discovered equations:")
model_undamped.print()

print("damped discovered equations")
model_damped.print()

# 4) Validate by simulation using the discovered models

sim_undamped = model_undamped.simulate(y0,t=t)
sim_damped = model_damped.simulate(y0,t=t)

# 5) Plot true vs predicted

fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

# Undamped x and x_pred
axs[0,0].plot(t, sol_undamped[:,0], label='x_true')
axs[0,0].plot(t, sim_undamped[:,0], '--' ,label='x_pred')
axs[0,0].set_title('Undamped: x(t)')
axs[0,0].legend()

# Undamped v and v_pred
axs[1,0].plot(t, sol_undamped[:,1], label='v_true')
axs[1,0].plot(t, sim_undamped[:,1], '--', label='v_pred')
axs[1,0].set_title('Undamped: v(t)')
axs[1,0].legend()

# damped x and x_pred
axs[0,1].plot(t, sol_damped[:,0], label='x_true')
axs[0,1].plot(t, sim_damped[:,0], '--', label='x_pred')
axs[0,1].set_title(f'Damped (c={damped_coeff}): x(t)')
axs[0,1].legend()

# damped v and v_pred
axs[1,1].plot(t, sol_damped[:,1], label='v_true')
axs[1,1].plot(t, sim_damped[:,1], '--', label='v_pred')
axs[1,1].set_title(f'Damped (c={damped_coeff}): v(t)')
axs[1,1].legend()

for ax in axs.flat:
    ax.set_xlabel('time')

plt.tight_layout()
plt.show()

