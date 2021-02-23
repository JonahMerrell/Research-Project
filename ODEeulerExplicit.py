import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
#https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html

Dv = 0.1 #a constant

x_min = 0.0
x_max = 1.0
x_count = 15 #15
dx = x_max/(x_count-1)
x_ = [x_min + j*dx for j in range(x_count)] #u[x][t]
x_[x_count-1] = x_max

t_min = 0.0
t_max = 2.0
t_count = 50 #50
dt = t_max/(t_count-1)
t_ =  [t_min + n*dt for n in range(t_count)] #u[x][t]
t_[t_count-1] = t_max

u=np.zeros((x_count,t_count))
g=np.zeros((x_count,t_count))

#apply initial condition u(x,0) = f(x) on u
for j in range(x_count):
    u[j ,0] = math.sin(x_[j] * math.pi)
    g[j, 0] = math.sin(x_[j] * math.pi)

_lambda = (dt) / (dx ** 2)
for n in range(t_count-1):
    u_predictor = np.zeros(x_count)#u[:,n]

    # Since our initial conditions u(0,t) = 0 and u(1,t) = 0, I just leave them as zero
    for j in range(1,x_count-1):
        u_predictor[j] = u[j, n] + Dv *(u[j - 1,n] - 2*u[j,n] + u[j + 1,n])
    for j in range(1, x_count - 1):
        u[j, n + 1] = u[j, n] + 0.5 * Dv * _lambda * ((u[j - 1, n] - 2 * u[j, n] + u[j + 1, n]) + (u_predictor[j-1] - 2*u_predictor[j] + u_predictor[j+1]))

        g[j, n + 1] = g[j,n] + Dv *_lambda * (g[j - 1,n] - 2*g[j,n] + g[j + 1,n])


#Everything below this is just for plotting u
x = np.linspace(x_min, x_max, x_count)
y = np.linspace(t_min, t_max, t_count)
X, Y = np.meshgrid(y, x)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, u, color='grey')
ax.plot_wireframe(X, Y, g, color='red')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('z');
plt.show()


