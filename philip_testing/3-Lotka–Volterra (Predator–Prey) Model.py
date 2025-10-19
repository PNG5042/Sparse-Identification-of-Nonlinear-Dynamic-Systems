import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# x˙=αx−βxy
# y˙​=δxy−γy

#Task:

#Simulate with odeint.
#Fit with PySINDy using polynomial features (degree 2).
#Interpret which terms it identifies.
#➡️ Bonus: Try to find what happens if you increase noise or reduce sampling rate.

