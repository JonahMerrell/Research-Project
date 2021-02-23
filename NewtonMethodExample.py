import numpy as np

#g=np.zeros((2,2))
J=np.zeros((2,2))
F=np.zeros(2)
u=np.zeros(2)
du=np.zeros(2)

u[0] = 1 #inital guess
u[1] = 2 #inital guess

for n in range(10):

    J[0,0] = 1
    J[0,1] = 2
    J[1,0] = 2*u[0]
    J[1,1] = 8*u[1]

    F[0] = u[0] + 2 * u[1] - 2  # = 0
    F[1] = u[0] ** 2 + 4 * u[1] ** 2 - 4  # = 0

    #The system of equations is J* dx = F
    du = np.linalg.solve(J, -F)
    u += du
print(u)
#Solution should be 0,1