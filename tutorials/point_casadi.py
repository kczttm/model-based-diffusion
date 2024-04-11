import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define the time horizon and number of control intervals
T = 4.0  # Time horizon
dt = 0.2  # Time step
N = int(T//dt)  # Number of control intervals
Ntraj = 10 # Number of trajectories to diffuse
Q = np.diag([0.5, 0.5])  # Weight matrix for the states
R = np.diag([0.1, 0.1])  # Weight matrix for the controls
x0 = np.array([-0.5, 0])  # Initial position
xf = np.array([0.5, 0])  # Final position
u_max = 1.0  # Maximum velocity
obs_center = np.array([[0.0, 0.0], [0.0, 0.5], [0.0, -0.5], [-0.5, -0.5], [-0.5, 0.5]])  # Center of the obstacle
obs_radius = 0.3  # Radius of the obstacle

def optimize_trajectory(x_guess, u_guess, yxs, yus, noise_var):
    # Define the optimization problem
    opti = ca.Opti()
    
    # Define the states and controls
    x = opti.variable(2, N+1)  # Position (x, y)
    u = opti.variable(2, N)  # Velocity (vx, vy)
    
    # Define the objective function
    objective = 0
    for k in range(N):
        # cost function
        objective += ca.mtimes([(x[:, k] - xf).T, Q, x[:, k] - xf]) + ca.mtimes([u[:, k].T, R, u[:, k]])
        # observation terms
        objective += ca.mtimes([(yxs[:, k] - x[:, k]).T, np.eye(2), yxs[:, k] - x[:, k]]) / noise_var
        objective += ca.mtimes([(yus[:, k] - u[:, k]).T, np.eye(2), yus[:, k] - u[:, k]]) / noise_var
    opti.minimize(objective)
    
    # Define the dynamic constraints
    for k in range(N):
        opti.subject_to(x[:, k+1] == x[:, k] + dt * u[:, k])  # Dynamics constraint
        opti.subject_to(u[:, k] <= u_max)  # Maximum velocity constraint
        opti.subject_to(u[:, k] >= -u_max)  # Minimum velocity constraint
    
    # Define the initial and final boundary conditions
    opti.subject_to(x[:, 0] == x0)
    opti.subject_to(x[:, -1] == xf)
    
    # Define the obstacle avoidance constraint

    for k in range(N+1):
        for i in range(obs_center.shape[0]):
            opti.subject_to(ca.dot(x[:, k] - obs_center[i], x[:, k] - obs_center[i]) >= obs_radius**2)
    
    # Set initial guess
    opti.set_initial(x, x_guess)
    opti.set_initial(u, u_guess)
    
    # Set the solver options
    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000, "tol": 1e-4}
    opti.solver("ipopt", p_opts, s_opts)
    
    # Solve the optimization problem
    sol = opti.solve()
    
    # Retrieve the optimized states and controls
    x_opt = sol.value(x)
    u_opt = sol.value(u)
    
    # Generate new observations
    yxs_new = x_opt + np.random.normal(0, np.sqrt(noise_var), (2, N+1))
    yus_new = u_opt + np.random.normal(0, np.sqrt(noise_var), (2, N))
    
    return x_opt, u_opt, yxs_new, yus_new

def plot_traj(ax, xss, uss, yxss, yuss):
    ax.clear()
    # obstacles
    for i in range(obs_center.shape[0]):
        circle = plt.Circle(obs_center[i, :], obs_radius, color='k', fill=True, alpha=0.5)
        ax.add_artist(circle)
    for j in range(Ntraj):
        x = xss[j]
        u = uss[j]
        ys = yxss[j]
        yu = yuss[j]
        # ax.plot(x[0, :], x[1, :], 'b-o', label='Optimized Trajectory')
        # ax.quiver(x[0, 1:], x[1, 1:], u[0, :], u[1, :], color='b')
        ax.plot(ys[0, :], ys[1, :], 'r-', label='Observations', alpha=0.1)
        ax.quiver(ys[0, :-1], ys[1, :-1], yu[0, :], yu[1, :], range(N), cmap="Reds")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    # ax.legend()
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Optimized Trajectory')

# Initialize the trajectory
def generate_xs():
    xmid = np.array([-1.0, (np.random.uniform(1.5, 2.5))*np.sign(np.random.uniform(-1.0, 1.0))])
    xs = np.concatenate([
        np.linspace(x0, xmid, N//2+1).T,
        np.linspace(xmid, xf, N-N//2).T
    ], axis=1)
    return xs
xss = np.array([generate_xs() for _ in range(Ntraj)])
us = np.zeros((2, N))
uss = np.tile(us, (Ntraj, 1, 1))
noise_init = 1.0
noise_end = 1e-3
noise_vars = np.exp(np.linspace(np.log(noise_init), np.log(noise_end), 30))
yxss = np.random.normal(0, np.sqrt(noise_vars[0]), (Ntraj, 2, N+1))
yxss[:, :, 0] = x0
yxss[:, :, -1] = xf
yuss = np.random.normal(0, np.sqrt(noise_vars[0]), (Ntraj, 2, N))

# Optimize the trajectory
fig, ax = plt.subplots()
for (i, noise_var) in enumerate(noise_vars):
    for j in range(Ntraj):
        xs, us, yxs, yus = optimize_trajectory(xss[j], uss[j], yxss[j], yuss[j], noise_var)
        xss[j] = xs
        uss[j] = us
        yxss[j] = yxs
        yuss[j] = yus
    # xs, us, yxs, yus = optimize_trajectory(xs, us, yxs, yus, noise_var)
    # Plot the optimized trajectory
    plot_traj(ax, xss, uss, yxss, yuss)
    plt.savefig(f"../figure/t_{i}.png")
    plt.pause(0.01)
plt.show()