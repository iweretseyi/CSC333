#!/usr/bin/env python
# coding: utf-8

# In[1]:


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



# In[2]:


# Problem 1: Maximize P = 3A + 4B
objective = [-3, -4]  # Negative for maximization
lhs_ineq = [[2, 3],   # 2A + 3B <= 12
            [1, 2]]   # A + 2B <= 8
rhs_ineq = [12, 8]
bounds = [(0, None), (0, None)]  # A, B >= 0

solve_lp(objective, lhs_ineq, rhs_ineq, bounds, "A (Product A)", "B (Product B)", "Maximizing Profit for a Factory")


# In[3]:


# Problem 2: Minimize C = 2X + 5Y
objective = [2, 5]  # Minimize
lhs_ineq = [[1, 2],   # X + 2Y <= 5
            [2, 1]]   # 2X + Y <= 6
rhs_ineq = [5, 6]
bounds = [(0, None), (0, None)]  # X, Y >= 0

solve_lp(objective, lhs_ineq, rhs_ineq, bounds, "X (Product X)", "Y (Product Y)", "Minimizing Cost for a Manufacturer")


# In[4]:


# Problem 3: Maximize P = 5A + 4B
objective = [-5, -4]  # Maximize
lhs_ineq = [[2, 1],   # 2A + B <= 20 (Labor)
            [3, 2],   # 3A + 2B <= 30 (Material)
            [1, 2]]   # A + 2B <= 18 (Machine time)
rhs_ineq = [20, 30, 18]
bounds = [(0, None), (0, None)]  # A, B >= 0

solve_lp(objective, lhs_ineq, rhs_ineq, bounds, "A (Product A)", "B (Product B)", "Maximizing Production with Multiple Resources")


# In[5]:


# Problem 4: Maximize R = 4A + 5B
objective = [-4, -5]  # Maximize
lhs_ineq = [[1, 2],   # A + 2B <= 15 (Production)
            [1, 2]]   # A + 2B <= 20 (Advertising)
rhs_ineq = [15, 20]
bounds = [(0, None), (0, None)]  # A, B >= 0

solve_lp(objective, lhs_ineq, rhs_ineq, bounds, "A (Product A)", "B (Product B)", "Maximizing Revenue from Sales")


# In[6]:


# Problem 5: Maximize P = 8P1 + 7P2
objective = [-8, -7]  # Maximize
lhs_ineq = [[3, 4],   # 3P1 + 4P2 <= 12 (Labor)
            [2, 1]]   # 2P1 + P2 <= 6 (Capital)
rhs_ineq = [12, 6]
bounds = [(0, None), (0, None)]  # P1, P2 >= 0

solve_lp(objective, lhs_ineq, rhs_ineq, bounds, "P1 (Project 1)", "P2 (Project 2)", "Resource Allocation for Two Projects")


# In[7]:


# Problem 6: Maximize P = 5C + 3V
objective = [-5, -3]  # Maximize
lhs_ineq = [[1, 2],   # C + 2V <= 8 (Baking time)
            [3, 2]]   # 3C + 2V <= 12 (Flour)
rhs_ineq = [8, 12]
bounds = [(0, None), (0, None)]  # C, V >= 0

solve_lp(objective, lhs_ineq, rhs_ineq, bounds, "C (Chocolate Cakes)", "V (Vanilla Cakes)", "Production Planning for a Bakery")


# In[8]:


# Problem 7: Minimize C = 6X + 7Y
objective = [6, 7]  # Minimize
lhs_ineq = [[3, 4],   # 3X + 4Y <= 18 (Fuel)
            [2, 1]]   # 2X + Y <= 10 (Driver time)
rhs_ineq = [18, 10]
bounds = [(0, None), (0, None)]  # X, Y >= 0

solve_lp(objective, lhs_ineq, rhs_ineq, bounds, "X (Vehicle X)", "Y (Vehicle Y)", "Minimizing Cost for a Transport Company")


# In[9]:


# Problem 8: Maximize R = 10P1 + 12P2
objective = [-10, -12]  # Maximize
lhs_ineq = [[4, 3],     # 4P1 + 3P2 <= 30 (Labor)
            [1, 2],     # P1 + 2P2 <= 18 (Raw materials)
            [3, 2]]     # 3P1 + 2P2 <= 24 (Machine time)
rhs_ineq = [30, 18, 24]
bounds = [(0, None), (0, None)]  # P1, P2 >= 0

solve_lp(objective, lhs_ineq, rhs_ineq, bounds, "P1 (Product 1)", "P2 (Product 2)", "Maximizing Revenue from Two Products")


# In[10]:


# Problem 9: Maximize R = 500000A + 400000B
objective = [-500000, -400000]  # Maximize
lhs_ineq = [[4000, 3000],   # 4000A + 3000B <= 5000 (TV budget)
            [2000, 2500],   # 2000A + 2500B <= 4500 (Print media budget)
            [1000, 1500]]   # 1000A + 1500B <= 3000 (Social media budget)
rhs_ineq = [5000, 4500, 3000]
bounds = [(0, None), (0, None)]  # A, B >= 0

solve_lp(objective, lhs_ineq, rhs_ineq, bounds, "A (Campaign A)", "B (Campaign B)", "Advertising Campaign Budget Allocation")


# In[11]:


# Problem 10: Maximize R = 6A + 5B
objective = [-6, -5]  # Maximize
lhs_ineq = [[2, 4],   # 2A + 4B <= 30 (Meat)
            [3, 2],   # 3A + 2B <= 24 (Vegetables)
            [1, 2]]   # A + 2B <= 20 (Rice)
rhs_ineq = [30, 24, 20]
bounds = [(0, None), (0, None)]  # A, B >= 0

solve_lp(objective, lhs_ineq, rhs_ineq, bounds, "A (Meal A)", "B (Meal B)", "Meal Planning for a School Cafeteria")


# In[ ]:




