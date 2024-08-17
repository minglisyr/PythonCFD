# Integration CFD work for
# 1. Incompressible
# 2. Laminar Flow
# 3. Transient
# 4. with Source Term
# ----------------------------------------------------------------------

##################
# Import section #
##################
import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

########################
# Fucntion definitions #
########################

## Periodic B.C. Building
def build_up_b(rho, dt, dx, dy, u, v):
    b = numpy.zeros_like(u)
    b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                      (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    
    # Periodic BC Pressure @ x = 2
    b[1:-1, -1] = (rho * (1 / dt * ((u[1:-1, 0] - u[1:-1,-2]) / (2 * dx) +
                                    (v[2:, -1] - v[0:-2, -1]) / (2 * dy)) -
                          ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx))**2 -
                          2 * ((u[2:, -1] - u[0:-2, -1]) / (2 * dy) *
                               (v[1:-1, 0] - v[1:-1, -2]) / (2 * dx)) -
                          ((v[2:, -1] - v[0:-2, -1]) / (2 * dy))**2))

    # Periodic BC Pressure @ x = 0
    b[1:-1, 0] = (rho * (1 / dt * ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx) +
                                   (v[2:, 0] - v[0:-2, 0]) / (2 * dy)) -
                         ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx))**2 -
                         2 * ((u[2:, 0] - u[0:-2, 0]) / (2 * dy) *
                              (v[1:-1, 1] - v[1:-1, -1]) / (2 * dx))-
                         ((v[2:, 0] - v[0:-2, 0]) / (2 * dy))**2))
    
    return b

## Pressure Poisson Building
def pressure_poisson_periodic(p, dx, dy):
    pn = numpy.empty_like(p)
    
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        # Periodic BC Pressure @ x = 2
        p[1:-1, -1] = (((pn[1:-1, 0] + pn[1:-1, -2])* dy**2 +
                        (pn[2:, -1] + pn[0:-2, -1]) * dx**2) /
                       (2 * (dx**2 + dy**2)) -
                       dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, -1])

        # Periodic BC Pressure @ x = 0
        p[1:-1, 0] = (((pn[1:-1, 1] + pn[1:-1, -1])* dy**2 +
                       (pn[2:, 0] + pn[0:-2, 0]) * dx**2) /
                      (2 * (dx**2 + dy**2)) -
                      dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 0])
        
        # Wall boundary conditions, pressure
        p[-1, :] =p[-2, :]  # dp/dy = 0 at y = 2
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
    
    return p

#############
# Main Body #
#############

## variable declarations
nx = 41
ny = 41
nt = 10
nit = 50 
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)
X, Y = numpy.meshgrid(x, y)


## physical variables
rho = 1
nu = .1
F = 1
dt = .01

## Initialize velocity fields
u = numpy.zeros((ny, nx))
v = numpy.zeros((ny, nx))

## Create a vortex-like initial condition
center_x, center_y = 1, 1  # Center of the domain
for i in range(ny):
    for j in range(nx):
        dx = X[i, j] - center_x
        dy = Y[i, j] - center_y
        r = numpy.sqrt(dx**2 + dy**2)
        theta = numpy.arctan2(dy, dx)
        if r > 0:
            u[i, j] = -numpy.sin(theta) * (1 - numpy.exp(-r/0.1))
            v[i, j] = numpy.cos(theta) * (1 - numpy.exp(-r/0.1))

un = u.copy()
vn = v.copy()

p = numpy.zeros((ny, nx))
pn = numpy.zeros((ny, nx))
b = numpy.zeros((ny, nx))

## Store the initial velocity for plotting
u_initial = u.copy()
v_initial = v.copy()

## Calculate initial velocity magnitude
initial_velocity_magnitude = numpy.sqrt(u**2 + v**2)

## iterations
udiff = 1
stepcount = 0
residuals = []

while udiff > .001 and stepcount < 1000:  # Added a maximum iteration limit
    un = u.copy()
    vn = v.copy()

    b = build_up_b(rho, dt, dx, dy, u, v)
    p = pressure_poisson_periodic(p, dx, dy)

    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * 
                    (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy * 
                    (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                     dt / (2 * rho * dx) * 
                    (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     nu * (dt / dx**2 * 
                    (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                     dt / dy**2 * 
                    (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) + 
                     F * dt)

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * 
                    (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy * 
                    (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                     dt / (2 * rho * dy) * 
                    (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                     nu * (dt / dx**2 *
                    (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                     dt / dy**2 * 
                    (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

    # Periodic BC for u and v
    u[:, -1] = u[:, 0]
    u[:, 0] = u[:, -2]
    v[:, -1] = v[:, 0]
    v[:, 0] = v[:, -2]

    # No-slip BC for top and bottom
    u[0, :] = 0
    u[-1, :] = 0
    v[0, :] = 0
    v[-1, :] = 0
    
    udiff = (numpy.sum(u) - numpy.sum(un)) / numpy.sum(u)
    stepcount += 1
    residuals.append(udiff)
    
    if stepcount % 10 == 0:
        print(f'Iteration {stepcount}, Residual = {udiff:.3e}')

print('After {} iterations, converged with a residual = {:.3e}'.format(stepcount, udiff))

# Create the figure with a 2x2 grid for the first 4 plots and an additional subplot for the 5th plot
fig = pyplot.figure(figsize=(16, 20))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.5])

# Plot 1: Initial Velocity Vector plot (top-left)
ax1 = fig.add_subplot(gs[0, 0])
vdense = 2
ax1.quiver(X[::vdense, ::vdense], Y[::vdense, ::vdense], u_initial[::vdense, ::vdense], v_initial[::vdense, ::vdense])
ax1.set_title('Initial Velocity Vector Field')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Plot 2: Initial velocity magnitude contour (top-right)
ax2 = fig.add_subplot(gs[0, 1])
initial_velocity_magnitude = numpy.sqrt(u_initial**2 + v_initial**2)
contour_initial = ax2.contourf(X, Y, initial_velocity_magnitude, levels=20, cmap='viridis')
ax2.set_title('Initial Velocity Magnitude Contour')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
fig.colorbar(contour_initial, ax=ax2, label='Initial Velocity Magnitude')
ax2.set_aspect('equal', adjustable='box')  # Make the aspect ratio square

# Plot 3: Final Vector plot (bottom-left)
ax3 = fig.add_subplot(gs[1, 0])
ax3.quiver(X[::vdense, ::vdense], Y[::vdense, ::vdense], u[::vdense, ::vdense], v[::vdense, ::vdense])
ax3.set_title('Final Velocity Vector Field')
ax3.set_xlabel('x')
ax3.set_ylabel('y')

# Plot 4: Final velocity magnitude contour (bottom-right)
ax4 = fig.add_subplot(gs[1, 1])
final_velocity_magnitude = numpy.sqrt(u**2 + v**2)
contour_final = ax4.contourf(X, Y, final_velocity_magnitude, levels=20, cmap='viridis')
ax4.set_title('Final Velocity Magnitude Contour')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
fig.colorbar(contour_final, ax=ax4, label='Final Velocity Magnitude')
ax4.set_aspect('equal', adjustable='box')  # Make the aspect ratio square

# Plot 5: Residual vs Iteration (bottom, spanning both columns)
ax5 = fig.add_subplot(gs[2, :])
ax5.semilogy(range(1, stepcount + 1), numpy.abs(residuals))
ax5.set_title('Residual vs Iteration')
ax5.set_xlabel('Iteration')
ax5.set_ylabel('Residual (log scale)')
ax5.grid(True)

pyplot.tight_layout()
pyplot.show()