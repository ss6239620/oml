import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 - 4*x + 4

def df(x):
    return 2*x - 4

def noisy_gradient_descent(learning_rate, initial_x, num_iterations, noise_level=0.1):
    x_values = [initial_x]
    for _ in range(num_iterations):
        grad = df(x_values[-1]) 
        noise = np.random.normal(0, noise_level)  
        noisy_grad = grad + noise  
        new_x = x_values[-1] - learning_rate * noisy_grad  
        x_values.append(new_x)
    return x_values

learning_rate = 0.1
initial_x = 0  
num_iterations = 50
noise_level = 1.0  

x_values = noisy_gradient_descent(learning_rate, initial_x, num_iterations, noise_level)

print("Noisy gradient descent steps:", x_values)

x = np.linspace(-1, 5, 100)
y = f(x)


plt.figure(figsize=(8, 6))
plt.plot(x, y, label='f(x) = x^2 - 4x + 4', color='b')  
plt.scatter(x_values, [f(x) for x in x_values], color='red', marker='x', label='Noisy Gradient Descent Steps')
plt.plot(x_values, [f(x) for x in x_values], color='r', linestyle='dashed', label='Noisy Descent Path')
plt.title('Noisy Gradient Descent Minimization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
