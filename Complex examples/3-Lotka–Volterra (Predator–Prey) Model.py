import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# x˙=αx−βxy
# y˙​=δxy−γy

#Task:
#Recover the nonlinear interaction terms xy

# Hints:
# Use PolynomialLibrary(degree=2)
# Visualize trajectories in the phase plane (x vs y)

alpha = 1.0
beta  = 0.5
delta = 0.5
gamma = 2.0

x0 = 2.0
y0 = 1.0

def derivative(X, t, alpha, beta, delta, gamma):
    x,y = X
    xdot = x * (alpha - beta * y)
    ydot = y * (-delta + gamma * x)
    return np.array([xdot, ydot])

t = np.linspace(0,30, 3001)

X0 = [x0, y0]

res = odeint(derivative, X0, t, args=(alpha, beta, delta, gamma))
x, y = res.T


