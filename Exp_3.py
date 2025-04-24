import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**4 - 4*x + 4

def df(x):
    return 4*x**3 - 4

def d2f(x):
    return 12*x**2 

def newtons_method(initial_x, num_iterations):
    x_values = [initial_x]
    for _ in range(num_iterations):
        grad = df(x_values[-1])  
        hess = d2f(x_values[-1])  
        
        new_x = x_values[-1] - grad / hess  
        x_values.append(new_x)

    return x_values


initial_x = 10  
num_iterations = 10  

x_values = newtons_method(initial_x, num_iterations)

print("Newton Method steps:", x_values)

x = np.linspace(0, 12, 100)  
y = f(x)


plt.figure(figsize=(8, 6))
plt.plot(x, y, label='f(x) = x^2 - 4x + 4', color='b')  
plt.scatter(x_values, [f(x) for x in x_values], color='red', marker='x', label='Newton Method Steps')
plt.plot(x_values, [f(x) for x in x_values], color='r', linestyle='dashed', label='Newton Path')

plt.title('Newton Method Minimization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
