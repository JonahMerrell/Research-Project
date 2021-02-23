import matplotlib.pyplot as plt

x_min = 0
x_max = 1.0
x_count = 50
dx = (x_max-x_min)/(x_count-1)
x_ = [x_min + j*dx for j in range(x_count)] #u[x][t]
x_[x_count-1] = x_max

y=[0 for i in range(x_count)]

def func(x, y):
    return (x + y + x * y) # dy / dx =(x + y + xy)
#apply initial condition y(0)
y[0] = 1.0
#Apply Euler's method
for j in range(x_count-1):
    y[j+1] = y[j] + dx * func(x_[j] , y[j])

#Everything below this is just for plotting y
plt.plot(x_,y,'r--', linewidth=2.0)
plt.xlabel("t")
plt.ylabel("S[N,C]")
plt.legend(["N","C"])
plt.show()
