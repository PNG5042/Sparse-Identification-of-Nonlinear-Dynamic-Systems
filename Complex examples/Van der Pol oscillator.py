import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pysindy as ps

mu = 2.0

def vdp(X, t):
    x, y = X
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]

t = np.linspace(0, 40, 4001)
X = odeint(vdp, [1.0, 0.0], t)

model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3))
model.fit(X, t=t[1]-t[0])
model.print()

X_sim = model.simulate(X[0], t)

plt.plot(X[:,0], X[:,1], label='True')
plt.plot(X_sim[:,0], X_sim[:,1], '--', label='SINDy')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(); plt.title('Van der Pol phase plane')
plt.show()
