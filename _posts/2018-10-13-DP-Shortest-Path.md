---
layout: post
title: Shortest Path Problem in a Grid
---

### Solving a common problem using recursive functions in Python
#### Introduction
When I got this assignment in one of my classes (Stochastic Dynamic Programming) I looked for example code online. My impression was that, for whatever reason, this problem is popularly solved in C++. After figuring out how to do it in Python I thought I'd share it here as an example code, mainly because I want to practice this whole github.io thing.
#### The Task
![Cost-Matrix](/images/Cost-Matrix.png "Cost-Matrix")
- The entries in the matrix above represent costs associated with the positions in the rectangle.
- We are interested in finding an optimal (min-cost) route from the top left-hand corner (origin) to the bottom
right-hand corner (destination).
- At each entry, we can go either to the *right* or *downwards*.
- The cost of a particular route is the sum of all the entries encountered on the way from the origin to the destination.
- What is the optimal route? 
#### Talking about the solution approach
The topic here is Dynamic Programming, which is a fancy term for "use smart recursive equations to solve your problem step by step. Everybody knows the elegant way to code a function in python that returns the factorial using a [recursive equation](https://www.python-course.eu/recursive_functions.php). The trick here is that we already know some trivial facts:
- The smaller the grid, the easier the problem. The smallest grid would be just one number, thus:
- The shortest path from the last grid entry to the last grid entry is the cost of the entry itself
- When we are on the last column (or row), we have no option but to move down (or right respectively). The task prohibits us from moving any other way (see above). So we just calculate them by adding up the numbers from the finish line backwards.
#### The non-trivial recursive equation
Using these trivial solutions, we have a well defined problem that we can unleash a powerful equation on: The shortest path from any given point is the minimum of the shortest path before, plus the cost of the cell. Now we translate that into Python code.

#### Here it is

```python
import numpy as np

# Solve Function
# matrix should be ndarray, srow is the start row index of the search (int), scolumn is the respective column
def sp(matrix, srow, scolumn):
    
    solution = 0
    
    # Smallest problem: shortest path from bottom right cell to bottom right cell is cost[bottom right cell].
    if srow == matrix.shape[0] and scolumn == matrix.shape[1]:
        solution = matrix[matrix.shape[0]-1][matrix.shape[1]-1]
        return solution
    
    # Define subproblems at the borders:
    elif srow == matrix.shape[0]:
        solution = sp(matrix, srow, scolumn + 1) + matrix[srow-1][scolumn-1]
        print("The current shortest path is %s for cell (%s, %s)" % (solution, srow, scolumn))
        return solution
    elif scolumn == matrix.shape[1]:
        solution = sp(matrix, srow + 1, scolumn) + matrix[srow-1][scolumn-1]
        print("The current shortest path is %s for cell (%s, %s)" % (solution, srow, scolumn))
        return solution
    
    # Recursive function for the rest of the problems:
    else:
        solution = min(sp(matrix, srow + 1, scolumn) + matrix[srow-1][scolumn-1],
                       sp(matrix, srow, scolumn + 1) + matrix[srow-1][scolumn-1])
        print("The current shortest path is %s for cell (%s, %s)" % (solution, srow, scolumn))
        return solution
```

You can test it on grids like these:

```python
# The grid to solve (bottom right cell is the final destination):
grid =  [[ 2, 5, 3, 8, 6],
         [ 4, 2, 9, 4, 4],
         [ 5, 3, 2, 6, 9],
         [ 0, 3, 8, 5, 0]]
cost = np.array(grid).reshape(len(grid), len(grid[0]))

# Execution example
sp(cost, 1, 1)
```

#### Note: 

This code is highly inefficient. In a second post I'll talk about why that is and how to fix that.
