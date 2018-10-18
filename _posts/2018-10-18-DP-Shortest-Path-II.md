---
layout: post
title: Dynamic Programming II
---

# Shortest Path Problem in a Grid - Revisited
### Solving a common problem using recursive functions in Python
#### Introduction
In my first post, I talked about a Python implementation that recursively solves a Shortest Path problem in a grid. At the end of that post I stated that the program would not be very efficient. To show why, here is the output of the console if you actually run the program:

 ```The current shortest path is 5 for cell (4, 4)
The current shortest path is 13 for cell (4, 3)
The current shortest path is 16 for cell (4, 2)
The current shortest path is 16 for cell (4, 1)
The current shortest path is 5 for cell (4, 4)
The current shortest path is 13 for cell (4, 3)
The current shortest path is 16 for cell (4, 2)
The current shortest path is 5 for cell (4, 4)
The current shortest path is 13 for cell (4, 3)
The current shortest path is 5 for cell (4, 4)
The current shortest path is 9 for cell (3, 5)
The current shortest path is 11 for cell (3, 4)
The current shortest path is 13 for cell (3, 3)
The current shortest path is 16 for cell (3, 2)
The current shortest path is 21 for cell (3, 1)
The current shortest path is 5 for cell (4, 4)
The current shortest path is 13 for cell (4, 3)
...
```
#### The Problem
All in all, the program produces around 90 lines of console output. But if we think about the original problem, we know that our 4 by 5 grid only has 20 cells. And because we already know the solution of the very last grid entry, we would only need to solve 19 subproblems.

![Cost-Matrix](/images/Cost-Matrix.png "Cost-Matrix")

Looking again at the output above, it seems that our program repeatedly solves the same subproblems. It tells us 4 times that the current shortest path for cell (4, 4) has a value of 5. 

Our problem here is that our program creates recursion stacks for new subproblems and has no way of knowing if it already solved a subproblem. Or another way to put it: after having solved a very easy problem at the start (say, "The current shortest path is 5 for cell (4, 4)", it does not make use of that result. Instead when having to calculate a slightly more advanced cell, like (3, 4), it solves all necessary contained subproblems again!
#### The Solution
The solution here is to make our program remember what it has already solved. We create a dictionary that contains the shortest paths' lenghts for every cell. We have to make it a global variable outside of the function, otherwise our program would forget that the dictionary had entries when it creates a new recursion stack.
#### The Code
```python
import numpy as np

# The grid to solve (bottom right cell is the final destination):
grid =  [[ 2, 5, 3, 8, 6],
         [ 4, 2, 9, 4, 4],
         [ 5, 3, 2, 6, 9],
         [ 0, 3, 8, 5, 0]]
cost = np.array(grid).reshape(len(grid), len(grid[0]))

# NEW: a memory to keep track of our solved paths, initialized with the smallest subproblem
memory = {(cost.shape[0],cost.shape[1]) : cost[cost.shape[0]-1,cost.shape[1]-1]}

# Solve Function
# matrix should be ndarray, srow is the start row index of the search (int), scolumn is the respective column
def sp(matrix, srow, scolumn):

    # Check if we already solved the subproblem
    if not (srow,scolumn) in memory:
        
        # Smallest problem: shortest path from bottom right cell to bottom right cell is cost[bottom right cell].
        # We initialized our memory with this solution
        if srow == matrix.shape[0] and scolumn == matrix.shape[1]:
            print("The current shortest path is %s for cell (%s, %s)" % (memory[srow,scolumn], srow, scolumn))

        # Define subproblems at the borders:
        elif srow == matrix.shape[0]:
            memory[srow,scolumn] = sp(matrix, srow, scolumn + 1) + matrix[srow-1][scolumn-1]
            print("The current shortest path is %s for cell (%s, %s)" % (memory[srow,scolumn], srow, scolumn))
        elif scolumn == matrix.shape[1]:
            memory[srow,scolumn] = sp(matrix, srow + 1, scolumn) + matrix[srow-1][scolumn-1]
            print("The current shortest path is %s for cell (%s, %s)" % (memory[srow,scolumn], srow, scolumn))

        # Recursive function for the rest of the problems:
        else:
            memory[srow,scolumn] = min(sp(matrix, srow + 1, scolumn) + matrix[srow-1][scolumn-1],
                                       sp(matrix, srow, scolumn + 1) + matrix[srow-1][scolumn-1])
            print("The current shortest path is %s for cell (%s, %s)" % (memory[srow,scolumn], srow, scolumn))
            # Just for deeper understanding -- uncomment if you want a messy output:
            #print(memory)
            
    return memory[srow,scolumn]
```

