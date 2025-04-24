import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

def solve_and_plot():
    c = [-3, -2]                             
    A_ub = [[1, 1],                           
            [2, 1]]
    b_ub = [4, 5]                             
    bounds = [(0, None), (0, None)]         

    result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                     bounds=bounds,
                     method='simplex')       

    if not result.success:
        raise RuntimeError("LP did not converge: " + result.message)

    x1_opt, x2_opt = result.x
    z_opt = -result.fun                    
    print(f"Optimal solution: x1 = {x1_opt:.4f}, x2 = {x2_opt:.4f}")
    print(f"Optimal objective value: {z_opt:.4f}")

    x = np.linspace(0, 4, 400)
    y1 = 4 - x
    y2 = 5 - 2 * x

    plt.plot(x, y1, label='$x_1 + x_2 = 4$')
    plt.plot(x, y2, label='$2x_1 + x_2 = 5$')
    plt.fill([0, 2.5, 1, 0], [0, 0, 3, 4],
             alpha=0.3, label='Feasible region')

    y_obj = (z_opt - 3 * x) / 2
    plt.plot(x, y_obj, '--', label=f'$3x_1+2x_2={z_opt:.0f}$')

    plt.scatter([x1_opt], [x2_opt], color='red', zorder=5,
                label=f'Optimum ({x1_opt:.2f},{x2_opt:.2f})')

    plt.xlim(0, 4)
    plt.ylim(0, 5)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('LP via SciPy Simplex')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    solve_and_plot()
