from random import randint, random, sample
from math import inf, sqrt
from itertools import accumulate

import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 



def create_graph(x, y):
    """calculate euclidean distances btw each nodes
    
    Rets
    ----
    (2D np.arr): distance matrix

    Example:
    [[0.         0.35693137 1.06066017 0.61008196]
     [0.35693137 0.         0.78892332 0.64899923]
     [1.06066017 0.78892332 0.         0.77278716]
     [0.61008196 0.64899923 0.77278716 0.        ]]
    """
    num_nodes = len(x)
    dist_matrix = np.zeros(shape=(num_nodes,num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            x1 = x[i]
            x2 = x[j]
            y1 = y[i]
            y2 = y[j]
            # euclidean dist calculation
            dist = sqrt((x1-x2)**2 + (y1-y2)**2)
            dist_matrix[i,j] = dist
    return dist_matrix


def draw_graph(x, y, node_size=50):
    """visualise all possible paths"""
    num_nodes = len(x)
    # draw edges
    for i in range(num_nodes-1):
        for j in range(num_nodes):
            x1 = x[i]
            y1 = y[i]
            x2 = x[j]
            y2 = y[j]
            X = [x1, x2]
            Y = [y1, y2]
            plt.plot(X, Y, "red", linewidth=0.5, alpha=0.9, zorder=1)
    # draw nodes
    plt.scatter(x, y, s=node_size, edgecolors="black", linewidth=0.5, zorder=2)


def draw_tour(x, y, best_tour, node_size=50):
    """draw best tour or least cost path"""
    num_nodes = len(best_tour)
    for i in range(num_nodes-1):
        current_node = best_tour[i]
        next_node = best_tour[i+1]
        x1 = x[current_node]
        y1 = y[current_node]
        x2 = x[next_node]
        y2 = y[next_node]
        X = [x1, x2]
        Y = [y1, y2]
        plt.plot(X, Y, "red", linewidth=1, zorder=1)

    # draw nodes
    plt.scatter(x, y, s=node_size, edgecolors="black", linewidth=0.5, zorder=2)


def draw_pheromone(x, y, tau, node_size=50):
    """draw pheromone trail levels"""
    min_tau = np.ndarray.min(tau)
    max_tau = np.ndarray.max(tau)

    tau_normalized = (tau - min_tau) / (max_tau - min_tau)

    num_nodes = len(x)
    # draw edges
    for i in range(num_nodes-1):
        for j in range(num_nodes):
            x1 = x[i]
            y1 = y[i]
            x2 = x[j]
            y2 = y[j]
            X = [x1, x2]
            Y = [y1, y2]
            plt.plot(X, Y, zorder=1, \
                    c=(0, 0, 1-tau_normalized[i][j], tau_normalized[i][j]), # RGBA
                    linewidth=5*tau_normalized[i][j],
                    alpha=0.9)

    # draw nodes
    plt.scatter(x, y, s=node_size, edgecolors="black", linewidth=0.5, zorder=2)


def plot_aco(x, y, best_tour, tau, node_size=50):
    """subplots of all edges/nodes, best tour, & pheromone path"""
    plt.rcParams.update({'font.size': 4})
    plt.rcParams["figure.figsize"] =(5,3)
    plt.rcParams['figure.dpi'] = 300

    plt.figure()
    ax1 = plt.subplot(1, 3, 1)
    draw_graph(x, y, node_size)
    
    ax2 = plt.subplot(1, 3, 2)
    draw_tour(x, y, best_tour, node_size)

    ax3 = plt.subplot(1, 3, 3)
    draw_pheromone(x, y, tau, node_size)

    ax1.set_title('All Nodes & Edges')
    ax2.set_title('Least Cost Trail')
    ax3.set_title('Pheromone Trail')
    plt.tight_layout()


def roulette_wheel(P):
    """choose a randomised value from a list based on roulette wheel selection;
    aka fitness proportion selection, commonly used in genetic algos.
    
    1. get cumulative sum of a list of probabilities
    2. generate a random float 0 < x < 1
    3. get first value in cumsum > random float
    4. match index back to list of probabilities to get the probability selected

    Rets
    ----
    (int): index (node) of probability being selected
    """
    cumsumP = list(accumulate(P))
    rand = random()

    for idx, i in enumerate(cumsumP):
        if i > rand:
            break
    return idx



def create_colony(graph, ant_no, tau, eta, alpha, beta):
    """get all paths based on pheromone & path cost"""
    num_nodes = len(graph)
    
    colony_tours = []
    # for each ant
    for i in range(ant_no):
        # select random initial node
        current_node = randint(0, num_nodes-1)
        ant_tour = []

        # to choose rest of nodes
        for j in range(num_nodes):
            ant_tour.append(current_node)
            # probability formula
            P_allnodes = (tau[current_node]**alpha) * (eta[current_node]**beta)
            # assign P = 0 to node already visited
            for i in ant_tour: P_allnodes[i] = 0
            P = P_allnodes / sum(P_allnodes)
            # choose next node
            current_node = roulette_wheel(P)
        # close the path
        ant_tour.append(ant_tour[0])
        colony_tours.append(ant_tour)
    colony_tours = np.array(colony_tours)
    return colony_tours


def fitness_func(tour, graph):
    """calculate the cost of all edges' weights for a single ant's tour"""
    fitness = 0
    for i in range(len(tour)-1):
        current_node = tour[i]
        next_node = tour[i+1]
        fitness += graph[current_node][next_node]
    return fitness


def update_pheromone(tau, colony_tours, fitness_colony):
    """update pheromone matrix tau, after one colony's iteration"""
    num_nodes = len(colony_tours[0])
    num_ants = len(colony_tours)
    
    for i in range(num_ants):
        tour = colony_tours[i]
        # for each node in the tour
        for j in range(num_nodes-1):
            current_node = tour[j]
            next_node = tour[j+1]

            tau[current_node][next_node] = tau[current_node][next_node] + 1/fitness_colony[i]
            tau[next_node][current_node] = tau[next_node][current_node] + 1/fitness_colony[i]
    return tau


def aco(x, y, maxiter, ant_no, rho=0.25, alpha=1, beta=1, display=True):
    """Ant Colony Optimization, a meta-heuristics algo for solving combinatorial problems
    Aim: find the shortest path that will tranverse pass all nodes (travelling salesman)
    Complexity: len(nodes)!, e.g. 5!=120, 10!=3628800 combinations
    Strategy: Stigmergy; ant paths built using pheromones trails

    Args
    ----
    x (list): of x coordinates
    y (list): of y coordinates
    maxiter: number of iterations for a colony to run
    ant_no: number of ants in a colony to run once
    rho (float): >0 & <1, evaporation rate (ρ), low=increase exploitation; high=increase exploration
    alpha (int/float): pheromone exponent parameter
    beta (int/float): edge cost (desirability) exponent parameter

    Rets
    ----
    (list): index of nodes to follow in sequence for shortest path
    """

    graph = create_graph(x, y)
    num_nodes = len(graph)
    mean_edges = np.mean(graph)

    # initial phermone for all edges
    tau0 = 10 * 1 / (num_nodes * mean_edges)
    # pheremone matrix, Τ
    tau = tau0 * np.ones((num_nodes, num_nodes))
    # desirability of an edge, η
    eta = 1 / graph


    fittest_overall = inf
    best_tour = []
    for i in range(maxiter):
        colony_tours = create_colony(graph, ant_no, tau, eta, alpha, beta)

        # calculate fitness value for each ant
        fitness_colony = []
        for tour in colony_tours:
            fitness = fitness_func(tour, graph)
            fitness_colony.append(fitness)

        # find fittest ant in the colony
        fittest_ant = min(fitness_colony)
        fittest_ant_idx = fitness_colony.index(fittest_ant)
        if fittest_ant < fittest_overall:
            fittest_overall = fittest_ant
            best_tour = colony_tours[fittest_ant_idx]
        
        # update pheromone matrix
        tau = update_pheromone(tau, colony_tours, fitness_colony)

        # evaporation pheromone
        tau = (1-rho) * tau

        # display results
        print("iteration: {}, best fitness: {}".format(i, fittest_overall))

        # plot
        if display:
            plot_aco(x, y, best_tour, tau)
            plt.pause(0.25)
            if i != maxiter-1:
                plt.close()
            else:
                plt.show()
                
    return list(best_tour)


if __name__ == "__main__":    
    x = sample(range(50), 50)
    y = sample(range(50), 50)
    maxiter = 20
    ant_no = 10
    display = True
    a = aco(x, y, maxiter, ant_no, rho=0.25, alpha=1, beta=1, display=display)
    print(a)