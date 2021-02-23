import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
#https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html

#Solves a system of differential equations, where both equations dependand on the other. (we solve for them both simultenously.

Du = 0.1 #a constant
Dv = 0.1 #a constant
c = 1.4 # a constant

x_min = 0; x_max = 1.0; x_count = 50
dx = (x_max-x_min)/(x_count-1)
x_ = [x_min + j*dx for j in range(x_count)] #u[x][t]
x_[x_count-1] = x_max

t_min = 0.0; t_max = 5.0; t_count = 50
dt = (t_max-t_min)/(t_count-1)
t_ =  [t_min + n*dt for n in range(t_count)] #u[x][t]
t_[t_count-1] = t_max

#Set up the inital variables
u=np.zeros((x_count,t_count))
v=np.zeros((x_count,t_count))
J=np.zeros((x_count*2,x_count*2))
F=np.zeros(x_count*2)


du=np.zeros(x_count)
dv=np.zeros(x_count)
#apply initial condition u(x,0) = f(x) on u
for j in range(x_count):
    u[j, 0] = math.sin(x_[j] * math.pi)
    v[j, 0] = 0.5*math.sin(x_[j] * math.pi)

_lambda = (dt) / (dx ** 2)

for n in range(t_count-1):

    u[:,n+1] = u[:,n] #Initial guess
    v[:, n + 1] = v[:, n]  # Initial guess
    for i in range(10):
        #Set up jacobian matrix for u
        for j in range(1,x_count-1):
            j2 = x_count+j
            J[j, j-1] = -2*Du * _lambda*u[j-1,n+1]
            J[j, j ] = 1 + 4 * Du * _lambda*u[j,n+1] - dt*c*v[j,n+1]
            J[j, j+1] = -2*Du * _lambda*u[j+1,n+1]

            J[j, j2] = -dt*c*u[j,n+1]

            J[j2, j2 - 1] = -1 * Dv * _lambda
            J[j2, j2] = 1 + 2 * Dv * _lambda - dt*c*u[j,n+1]
            J[j2, j2 + 1] = -1 * Dv * _lambda

            J[j2, j] = -dt*c*v[j, n + 1]

        # Set up F Vector for entire system.
        for j in range(1, x_count - 1):
            F[j] = u[j,n+1] - u[j, n] - Du*_lambda*(u[j-1,n+1]**2 - 2*u[j,n+1]**2 + u[j+1,n+1]**2)  - dt*c*u[j,n+1]*v[j,n+1]
            F[x_count + j] = v[j,n+1] - v[j, n] - Dv*_lambda*(v[j-1,n+1] - 2*v[j,n+1] + v[j+1,n+1]) - dt*c*u[j,n+1]*v[j,n+1]

        # Remove the boundry conditions columns/rows of J and F, since they are already known
        J_temp = np.delete(J, [0, x_count - 1,x_count,2*x_count-1], 0)
        J_temp = np.delete(J_temp, [0, x_count - 1,x_count,2*x_count-1], 1)
        F_temp = np.delete(F, [0, x_count - 1,x_count,2*x_count-1])


        #Solve the system J*du = -F for du (according to Newtons method), then add du to u[:,n+1] to converge closer to solution
        du = np.linalg.solve(J_temp, -F_temp)
        l = len(du)//2

        u[:,n+1] += np.hstack([0, du[:l], 0])
        v[:,n+1] += np.hstack([0, du[l:], 0])

#Everything below this is just for plotting u and v
x = np.linspace(x_min, x_max, x_count)
y = np.linspace(t_min, t_max, t_count)
X, Y = np.meshgrid(y, x)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, u, color='grey')
ax.plot_wireframe(X, Y, v, color='red')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('z');
plt.show()


