import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math


Dv = 0.1 #a constant

x_min = 0; x_max = 1.0; x_count = 50
dx = (x_max-x_min)/(x_count-1)
x_ = [x_min + j*dx for j in range(x_count)] #u[x][t]
x_[x_count-1] = x_max

t_min = 0.0; t_max = 2.0; t_count = 50
dt = (t_max-t_min)/(t_count-1)
t_ =  [t_min + n*dt for n in range(t_count)] #u[x][t]
t_[t_count-1] = t_max

u=np.zeros((x_count,t_count))
J=np.zeros((x_count,x_count))
F=np.zeros(x_count)
du=np.zeros(x_count-2)
#apply initial condition u(x,0) = f(x) on u
for j in range(x_count):
    u[j, 0] = math.sin(x_[j] * math.pi)
#Techniqually, we would now apply our boundry conditions on u here, but since they're 0 in this case, there is no need.



_lambda = (dt) / (dx ** 2)

for n in range(t_count-1):
    u[:,n+1] = u[:,n] #Initial guess
    for i in range(10):
        #Set up jacobian matrix
        for j in range(1, x_count-1):#Remember that J is actaully size x_count-2, (since we already know the boundry conditions), hence we "cut off" the edges of the loop range.
            J[j, j-1] = -2*Dv * _lambda*u[j-1,n+1]
            J[j, j  ] = (1 + 4 * Dv * _lambda*u[j,n+1])
            J[j, j+1] = -2*Dv * _lambda*u[j+1,n+1]

        # Set up F Vector
        for j in range(1, x_count - 1):
            F[j] = u[j,n+1]-u[j, n] - Dv * _lambda*(u[j-1,n+1]**2 - 2*u[j,n+1]**2 + u[j+1,n+1]**2)

        #Remove the boundry conditions columns/rows of J and F, since they are already known
        J_temp = np.delete(J, [0,x_count - 1], 0)
        J_temp = np.delete(J_temp, [0,x_count - 1], 1)
        F_temp = np.delete(F, [0,x_count - 1])

        #Solve the system J*du = -F for du (according to Newtons method), then add du to u[:,n+1] to converge closer to solution
        du = np.linalg.solve(J_temp, -F_temp)
        u[:,n+1] += np.hstack([0, du, 0])#insert the boundry condition rows into the matrix. (we add 0s to them since we already know our boundry condtions are correct)


#Everything below this is just for plotting u
x = np.linspace(x_min, x_max, x_count)
y = np.linspace(t_min, t_max, t_count)
X, Y = np.meshgrid(y, x)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, u, color='grey')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('z');
plt.show()


