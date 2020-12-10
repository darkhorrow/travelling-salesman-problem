#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple

import networkx as nx
from collections import OrderedDict
import time
import random
import copy

Point = namedtuple("Point", ['x', 'y'])
points = []


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def cost(path):
    cost = 0
    for node, next_node in zip(path, path[1:]):
        cost += length(points[node], points[next_node])
    return cost


def invert_list(list, origin, destination):
    i = origin
    j = destination
    old_cost = cost([list[i - 1], list[i]]) + cost([list[j], list[j + 1]])
    new_cost = cost([list[i - 1], list[j]]) + cost([list[i], list[j + 1]])
    if new_cost < old_cost:
        list[i:j + 1] = reversed(list[i:j + 1])
        return True
    return False


def calc_temp(delta, h):
    return -delta / math.log(h)


def approximation2(g):
    return list(nx.dfs_preorder_nodes(nx.minimum_spanning_tree(g), 1))


def christofides(g):
    t = nx.MultiGraph(nx.minimum_spanning_tree(g))
    w = nx.Graph()
    for (node, degree) in t.degree():
        if degree % 2 != 0:
            w.add_node(node)
    for n1 in w:
        for n2 in w:
            if n1 != n2:
                w.add_edge(n1, n2, weight=-length(points[n1], points[n2]))
    m = nx.max_weight_matching(w, maxcardinality=True)
    for (n1, n2) in m:
        t.add_edge(n1, n2, weight=length(points[n1], points[n2]))
    c = nx.eulerian_circuit(t)
    h = [next(c)[0]]
    for pair in c:
        h.append(pair[1])
    h = list(OrderedDict.fromkeys(h).keys())
    h.append(0)
    return h


def opt2(g):
    feasible = christofides(g)
    try_again = True
    while try_again:
        try_again = False
        for i in range(1, len(feasible) - 1):
            for j in range(i + 1, len(feasible) - 1):
                try_again = invert_list(feasible, i, j)
    return feasible


def simulated_annealing(g, alpha=0.995, h=0.9):
    random.seed(30)
    current = christofides(g)
    ini_cost = 0
    for node, next_node in zip(current, current[1:]):
        if cost([node, next_node]) > ini_cost:
            ini_cost = cost([node, next_node])
    delta = ini_cost/10
    t = calc_temp(delta, h)
    t_med = calc_temp(delta, 0.4)
    t_med2 = calc_temp(delta, 0.2)
    t_final = calc_temp(delta, 1E-20)
    max_iter = 10
    n = 0
    flagm = False
    flagm2 = False
    while t >= t_final:
        while n < max_iter:
            i = random.randint(1, len(current) - 3)
            j = random.randint(i + 2, len(current) - 1)
            old_cost = length(points[current[i - 1]], points[current[i]]) + length(points[current[j - 1]],
                                                                                   points[current[j]])
            new_cost = length(points[current[i - 1]], points[current[j]]) + length(points[current[j - 1]],
                                                                                   points[current[i]])
            delta = new_cost - old_cost
            if delta < 0:
                current[i:j] = reversed(current[i:j])
            elif random.random() < math.exp(-delta / t):
                current[i:j] = reversed(current[i:j])
            n = n + 1
        n = 0
        t = alpha * t
        if t < t_med and not flagm:
            max_iter = 8*len(current)
            flagm = True
        if t < t_med2 and not flagm2:
            max_iter = 17*len(current)
            flagm2 = True
    return current


def tabu_search(g):
    feasible = christofides(g)
    max_iter = 200
    count = 1
    size_tabu = len(g)
    tabu = []
    dropper = cost(feasible)
    best_dropper = dropper
    best_circuit = copy.deepcopy(feasible)
    while count <= max_iter:
        best_change = -math.inf
        best_pair = []
        for i in range(1, len(feasible)-3):
            for j in range(i + 2, len(feasible)-1):
                if (feasible[i], feasible[j]) in tabu:
                    continue
                else:
                    old_cost = length(points[feasible[i]], points[feasible[i+1]]) + length(points[feasible[j]],
                                                                                           points[feasible[j+1]])
                    new_cost = length(points[feasible[i]], points[feasible[j]]) + length(points[feasible[i+1]],
                                                                                         points[feasible[j+1]])
                    local_cost = old_cost - new_cost
                    if local_cost > best_change:
                        best_change = local_cost
                        best_pair = [i, j]
        dropper -= best_change
        feasible[best_pair[0]+1:best_pair[1]+1] = reversed(feasible[best_pair[0]+1:best_pair[1]+1])
        if dropper < best_dropper:
            best_dropper = dropper
            best_circuit = copy.deepcopy(feasible)
        tabu.insert(0, (feasible[best_pair[0]], feasible[best_pair[1]]))
        if len(tabu) > size_tabu:
            tabu.pop()
        count += 1
    return best_circuit


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    node_count = int(lines[0])

    global points
    g = nx.Graph()
    for i in range(1, node_count + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))
        g.add_node(i - 1)
    for i in range(node_count):
        for j in range(node_count):
            if i != j:
                g.add_edge(i, j, weight=length(points[i], points[j]))
    t = time.time()
    solution = tabu_search(g)  # Tipo de solución, cambiar para ejecutar un algoritmo u otro
    print('Tiempo: ' + str(time.time() - t))
    # prepare the solution in the specified output format
    output_data = '%.2f' % cost(solution) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. '
              '(i.e. python solver.py ./data/tsp_51_1)')
