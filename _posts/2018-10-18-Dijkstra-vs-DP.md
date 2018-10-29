---
layout: post
title: Dijkstra versus Dynamic Programming
---

### Similarities between the recursive formulation and Dijkstra's Algorithm in Digraphs with positive arc weights
#### Dijkstra's Algorithm:
Dijkstra's Algorithm is a popular and widely used method in optimization, as it efficiently calculates a shortest path in any connected Digraph with positive arc weights. At its core stands the idea that after having found the shortest path to a node, this result is stored in the list of "permanent" nodes and its value can be used for further calculations.

This is possible because of the exclusive use of positive (or zero value) arc weights. Calculated shortest paths do not need to be corrected.
A "pseudo-code" for this algorithm can be written as follows:

![Pseudo-Code](/images/Pseudo-Code.png "Pseudo-Code")

***

#### Recursive equation for the Problem:
A (seemingly) different approach to solve this problem is with a recursive equation based in the discipline of dynamic programming: decompose the problem into smaller sub problems with similar structure, and use the (often trivial) solution of the smallest sub problem to consecutively solve larger and larger problems, until one arrives at the original question.

This approach can be written as follows:

![DP-Dijkstra](/images/DP-Dijkstra.png "DP-Dijkstra")

***

#### The connection between the two:

Although the two approaches are based off two different ideas they share a lot of connections: The Dijkstra Algorithm - although not immediately apparent - is built on the principle of optimality, which is used in every problem of dynamic programming. In its stepwise search it makes use of the assumption that a shortest path is in itself a combination of shortest paths. Thus, in reality, the biggest difference is just the sequence by which we try to solve this problem: The propagation of the Dijkstra algorithm (DA) is based on nodes, the steps of the recursive equation are based on arcs.

The similarities begin with the initialization of the DA (lines 5 - 10 of the pseudo code). We define the distance of the starting node to itself to be zero. The distance to the rest of the nodes is set to be infinity infinity. This is similar to equation (3) in the recursive formulation. These trivial solutions represent the starting point of our calculations.

The core similarity however is between line 12-21 and equation (4): At every step of our calculations we look at a certain node (u in the pseudo code or v respectively) and its neighbors. The section min((j,i)∈A)⁡{v_(k-1) (j)+ d_ji } in (4) bears striking similarity to head of the for-loop (line 17 - 18). We go through the neighbors of the node we are interested in and calculate the distances. If we find a value "alt" that is smaller than our originally found shortest path for our node (line 19), meaning that we found min((j,i)∈A)⁡{v_(k-1) (j)+ d_ji } to be smaller than v_(k-1) (i), we save it as the new shortest path. Thus we are using the minimum of the two expressions. This procedure is analogous to the first min-expression of equation (4). In conclusion these statements mean that the core functionalities of the two algorithms are the same: at every step we look for possible new shortest paths using the adjacent nodes, but do not overwrite already found shortest paths.

The rest of the lines of the pseudo code in the Dijkstra basically handles our data: the unvisited nodes (lines 12 - 13), the predecessors (line 21), and the distances for each node (line 20) are altered and saved for each iteration of the for-loop. These statements are only implicitly found in the recursive equations: When we find a solution using the recursion we would need to back trace our calculations to find our shortest path.
