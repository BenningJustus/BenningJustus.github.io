---
layout: post
title: Dynamic Programming
---

# Shortest Path Problem in a Grid
### Solving a common problem using recursive functions in Python

When I got this assignment in one of my classes (Stochastic Dynamic Programming) I looked for example code online. My impression was that, for whatever reason, this problem is popularly solved in C++. After figuring out how to do it in Python I thought I'd share it here as an example code, mainly because I want to practice this whole github.io thing.

#### Here it is:

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

#### A few notes and comments:

This code is highly inefficient. In a second Post I'll talk about why that is and how to fix that.
