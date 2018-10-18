---
layout: post
title: Dijkstra vs. Dynamic Programming
published: false
---

### Similarities between the recursive formulation of the shortest path problem and Dijkstra's Algorithm in Digraphs with positive arc weights 

#### Dijkstra's Algorithm:
Dijkstra's Algorithm is a popular and widely used method in optimization, as it efficiently calculates a shortest path in any connected Digraph with positive arc weights. At its core stands the idea that after having found the shortest path to a node, this result is stored in the list of "permanent" nodes and its value can be used for further calculations. This is possible because of the exclusive use of positive (or zero value) arc weights. Calculated shortest paths do not need to be corrected.
A "pseudo-code" for this algorithm can be written as follows:

![Pseudo-Code](/images/Pseudo-Code.png "Pseudo-Code")

#### Recursive equation for the Problem:
A (seemingly) different approach to solve this problem is with a recursive equation based in the discipline of dynamic programming: decompose the problem into smaller sub problems with similar structure, and use the (often trivial) solution of the smallest sub problem to consecutively solve larger and larger problems, until one arrives at the original question.

This approach can be written as follows: 

![DP-Dijkstra](/images/DP-Dijkstra.png "DP-Dijkstra")

