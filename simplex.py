import numpy as np
import matplotlib.pyplot as plt

def simplex(Z, X, Rhs):
    Z = np.array(Z, dtype=float)
    X = np.array(X, dtype=float)
    Rhs = np.array(Rhs, dtype=float)
    
    num_constraints, num_variables = X.shape

    tableau = np.zeros((num_constraints + 1, num_variables + num_constraints + 1))

    tableau[:-1, :-1] = np.hstack((X, np.eye(num_constraints)))  
    tableau[:-1, -1] = Rhs  
    tableau[-1, :-1] = np.hstack((-Z, np.zeros(num_constraints)))  

    while np.any(tableau[-1, :-1] < 0):  
        pivot_col = np.argmin(tableau[-1, :-1])
        
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf  
        pivot_row = np.argmin(ratios)
        
        if np.all(ratios == np.inf):
            raise ValueError("Problem is unbounded.")
        
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
        
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
    
    solution = np.zeros(num_variables)
    for i in range(num_variables):
        col = tableau[:-1, i]
        if np.count_nonzero(col) == 1 and np.sum(col) == 1: 
            row = np.where(col == 1)[0][0]
            solution[i] = tableau[row, -1]
    
    optimal_value = tableau[-1, -1]  
    
    return optimal_value, solution

def plot_graph(X, Rhs, solution):
    x = np.linspace(0, 10, 200)
    
    y1 = (Rhs[0] - X[0][0] * x) / X[0][1]
    y2 = (Rhs[1] - X[1][0] * x) / X[1][1]
    
    plt.figure(figsize=(8,6))
    plt.plot(x, y1, label=f"{X[0][0]}x + {X[0][1]}y <= {Rhs[0]}")
    plt.plot(x, y2, label=f"{X[1][0]}x + {X[1][1]}y <= {Rhs[1]}")
    
    plt.fill_between(x, np.minimum(y1, y2), 0, where=(np.minimum(y1, y2) >= 0), alpha=0.3)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.scatter(solution[0], solution[1], color='red', zorder=3, label='Optimal Solution')
    plt.title("Feasible Region and Optimal Solution")
    plt.grid()
    plt.legend()
    plt.show()

Z = [5, 4]  
X = [
    [6, 4],  
    [1, 2]   
]
Rhs = [24, 6]  

optimal_value, solution = simplex(Z, X, Rhs)
print("Optimal Value:", optimal_value)
print("Solution:", solution)

plot_graph(X, Rhs, solution)
