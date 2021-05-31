# Heuristic Algorithms

Implementations of some optimization algorithms like Ant Colony Optimization.

## Ant Colony Optimization

This algorithm is used to solve the travelling salesman problem. To visualize the best path & pheromone trail per iteration we can select `display = True`. However, the script will run much slower.

```python
from aco import aco
from random import sample

# create 20 nodes
x = sample(range(20), 20)
y = sample(range(20), 20)

maxiter = 20
ant_no = 10
display = True
a = aco(x, y, maxiter, ant_no, rho=0.25, alpha=1, beta=1, display=display)
```

<p>
    <img src="https://github.com/mapattacker/heuristic-algorithms/blob/master/img/aco.png?raw=true" width=60% />
</p>