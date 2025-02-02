import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def integrator_example_loop(t, state, params):
    # Initialize the state vector derivative
    state_dot = np.zeros_like(state)
    # Populating the state derivative
    state_dot[0] = -params["decay_rate_x"] * state[0]
    state_dot[1] = -params["decay_rate_y"] * state[1]
    state_dot[2] = -params["decay_rate_z"] * state[2]
    # Returning state vector derivative
    return state_dot


initial_state = np.array([10, -20, 30])
t_span = (0, 500)
params = {
    "decay_rate_x": 0.01,
    "decay_rate_y": 0.025,
    "decay_rate_z": 0.015,
}
n_steps = 100
result = solve_ivp(
    integrator_example_loop,
    t_span,
    initial_state,
    method="RK45",
    t_eval=np.linspace(t_span[0], t_span[1], n_steps),
    args=(params,),
)

# Extracting the results
t = result.t
state_vector = result.y
# Plotting the output
plt.plot(t, state_vector[0, :], label=r"$x$")
plt.plot(t, state_vector[1, :], label=r"$y$")
plt.plot(t, state_vector[2, :], label=r"$z$")
plt.xlabel(r"Time (s)")
plt.ylabel(r"State")
plt.grid("on")
plt.legend()
plt.savefig("Assignment 2/RungeKutta45_test.png")
plt.show()
