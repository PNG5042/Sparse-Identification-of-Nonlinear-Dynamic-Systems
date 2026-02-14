import numpy as np
import pysindy as ps


def test_pysindy_linear_system():
    # Generate simple exponential decay data
    t = np.linspace(0, 5, 100)
    x = np.exp(-t)

    x = x.reshape(-1, 1)

    # Create SINDy model
    model = ps.SINDy()

    # Fit model
    model.fit(x, t=t)

    # Get identified coefficients
    coefficients = model.coefficients()

    # Expect coefficient close to -1
    learned_value = coefficients[0][1]

    assert np.isclose(learned_value, -1, atol=0.2)
