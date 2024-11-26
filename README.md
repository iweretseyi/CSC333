# CSC333
linear programming 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

def solve_lp(objective, lhs_ineq, rhs_ineq, bounds, x_label, y_label, title):
    # Set up grid for graphing
    x = np.linspace(0, 100, 500)  # Adjust the range if needed
    
    # Create constraint lines
    y_constraints = []
    for coef, rhs in zip(lhs_ineq, rhs_ineq):
        if coef[1] != 0:  # Avoid division by zero for vertical lines
            y_constraints.append((rhs - coef[0]*x) / coef[1])
        else:
            y_constraints.append(np.full_like(x, rhs / coef[0]))
    
    # Plot constraints
    plt.figure(figsize=(8, 6))
    for i, y in enumerate(y_constraints):
        plt.plot(x, y, label=f"Constraint {i+1}")
    
    # Shade feasible region (gray area)
    y_min = np.minimum.reduce([np.maximum(0, y) for y in y_constraints])
    plt.fill_between(x, 0, y_min, where=(y_min >= 0), color='gray', alpha=0.3)
    
    # Labels and formatting
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid()
    plt.xlim(0, max(x))
    plt.ylim(0, 20)  # Adjust range based on problem
    plt.show()
    
    # Optimization using scipy
    result = linprog(c=objective, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bounds, method="highs")
    print("Optimal Solution:")
    print(f"x1 = {result.x[0]:.2f}, x2 = {result.x[1]:.2f}")
    print(f"Optimal Value = {(-result.fun if objective[0] < 0 else result.fun):.2f}\n")

