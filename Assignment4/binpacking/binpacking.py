#! /usr/bin/env python3
"""NAMES OF THE AUTHOR(S): Gaël Aglin <gael.aglin@uclouvain.be>"""
import time

from search import *
from copy import deepcopy
import sys
import random

def fullness(state, id_bin):
    return sum(state.bins[id_bin].values())

def item_fits(state, id_bin, size):
    return (state.capacity - fullness(state,id_bin)) >= size

def is_empty_bin(state, id_bin):
    return len(state.bins[id_bin]) == 0

def which_bin_item(state, id_item):
    for id_bin in range(len(state.bins)):
        if id_item in state.bins[id_bin].keys():
            return id_bin

def empty_space(state, id_bin):
    return state.capacity - fullness(state,id_bin)

def one_item_swaps(state):
    successors = []
    for x, bin_x in enumerate(state.bins):
        for y, bin_y in enumerate(state.bins):
            if x != y:
                for id_item, size in bin_x.items():
                    if item_fits(state, y, size):
                        new_state = deepcopy(state)
                        new_state.bins[x].pop(id_item)
                        new_state.bins[y][id_item] = size

                        if is_empty_bin(state,x):
                            del new_state.bins[x]

                        successors.append(new_state)

    return successors

def two_items_swaps(state):
    successors = []
    for x, bin_x in enumerate(state.bins):
        for y, bin_y in enumerate(state.bins):
            if x < y:
                for id_item_x, size_x in bin_x.items():
                    for id_item_y, size_y in bin_y.items():
                        if size_x != size_y:
                            new_state = deepcopy(state)
                            new_state.bins[x].pop(id_item_x)
                            new_state.bins[y].pop(id_item_y)

                            if item_fits(new_state, x, size_y) and item_fits(new_state, y, size_x):
                                new_state.bins[x][id_item_y] = size_y
                                new_state.bins[y][id_item_x] = size_x
                                successors.append(new_state)

    return successors

class BinPacking(Problem):
    def successor(self, state):
        for suc in one_item_swaps(state):
            yield "one", suc

        for suc in two_items_swaps(state):
            yield "two", suc

    def fitness(self, state):
        nb_bins = len(state.bins)
        capacity = state.capacity

        score = 0
        for i in range(nb_bins):
            score += (fullness(state, i) / capacity) ** 2

        score /= nb_bins
        return 1 - score

class State:
    def __init__(self, capacity, items):
        self.capacity = capacity
        self.items = items
        self.bins = self.build_init()

    # an init state building is provided here but you can change it at will
    def build_init(self):
        init = []
        for ind, size in self.items.items():
            if len(init) == 0 or not self.can_fit(init[-1], size):
                init.append({ind: size})
            else:
                if self.can_fit(init[-1], size):
                    init[-1][ind] = size
        return init

    def can_fit(self, bin, itemsize):
        return sum(list(bin.values())) + itemsize <= self.capacity

    def __str__(self):
        s = ''
        for i in range(len(self.bins)):
            s += ' '.join(list(self.bins[i].keys())) + '\n'
        return s


def read_instance(instanceFile):
    file = open(instanceFile)
    capacitiy = int(file.readline().split(' ')[-1])
    items = {}
    line = file.readline()
    while line:
        items[line.split(' ')[0]] = int(line.split(' ')[1])
        line = file.readline()
    return capacitiy, items

# Attention : Depending of the objective function you use, your goal can be to maximize or to minimize it
def maxvalue(problem, limit=100, callback=None):
    current = LSNode(problem, problem.initial, 0)

    best = current

    global_cpt = 0
    global_max = math.inf
    global_time_start = time.time()
    global_time = 0

    for step in range(limit):
        if callback is not None:
            callback(current)

        max_state = None
        max_fitness = math.inf
        successors = list(best.expand())

        for suc in successors:
            next_state = suc.state
            next_fitness = problem.fitness(next_state)

            if next_fitness < max_fitness:
                max_state = next_state
                max_fitness = next_fitness

        if max_fitness < global_max:
            global_max = max_fitness
            global_cpt = step
            global_time = time.time() - global_time_start

        best = LSNode(problem, max_state, step + 1)

    best.max_fit = global_max
    best.max_limit = global_cpt + 1
    best.max_time = global_time

    return best

# Attention : Depending of the objective function you use, your goal can be to maximize or to minimize it
def randomized_maxvalue(problem, limit=100, callback=None):
    current = LSNode(problem, problem.initial, 0)
    best = current

    global_cpt = 0
    global_max = math.inf
    global_time_start = time.time()
    global_time = 0

    for step in range(limit):
        if callback is not None:
            callback(current)

        successors = list(best.expand())
        successors.sort(key = lambda x: problem.fitness(x.state))

        rand = random.randint(0,4)
        best = LSNode(problem, successors[rand].state, step + 1)

        best_state = successors[rand].state
        if problem.fitness(best_state) < global_max:
            global_max = problem.fitness(best_state)
            global_cpt = step
            global_time = time.time() - global_time_start

    best.max_fit = global_max
    best.max_limit = global_cpt + 1
    best.max_time = global_time

    return best

#####################
#       Launch      #
#####################
if __name__ == '__main__':
    ### For INGINIOUS ###
    # info = read_instance(sys.argv[1])
    for i in range(1,11):
        info = read_instance("instances/i0"+str(i)+".txt")
        init_state = State(info[0], info[1])
        bp_problem = BinPacking(init_state)

        mean_time,mean_fit,mean_limit = 0,0,0
        node = None
        j = 0
        for j in range(10):

            step_limit = 100
            choose = 2
            start = time.time()

            if choose == 0:
                node = maxvalue(bp_problem, step_limit)
            elif choose == 1:
                node = randomized_maxvalue(bp_problem, step_limit)
            else:
                node = random_walk(bp_problem, step_limit)

            time_now = time.time() - start
            mean_time += node.max_time
            mean_fit += node.max_fit
            mean_limit += node.max_limit

        j+=1
        mean_time /= j
        mean_fit /= j
        mean_limit /= j

        # print(str(i) + " & " + str(round(node.best_time*1000)) + " & " + str(node.max_fit) + " & " + str(node.limit))
        print(str(i) + " & " + str(round(mean_time*1000)) + " & " + str(round(mean_fit,4)) + " & " + str(round(mean_limit,1)))
        # print(node.state)
        # print("T(s) : " + str(time_now))
        # print("Fitness : " + str(bp_problem.fitness(node.state)))
        # print("Nb. Steps : " + str(node.step))