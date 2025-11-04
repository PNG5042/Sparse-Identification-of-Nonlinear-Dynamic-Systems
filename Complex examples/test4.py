import numpy as np
import pysindy as py
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#define the ODE
def logistic_growth(x,t):
    return (x * (1-x))

#Create time points
t = np.linspace(0, 10, 100)

x0 = 0.1

# Step 4: Integrate the ODE
x = odeint(logistic_growth, x0, t)

# Step 5: Plot the result
plt.plot(t, x)
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.title('Logistic Growth')
plt.show()