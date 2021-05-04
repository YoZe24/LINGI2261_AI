#! /usr/bin/env python3
"""NAMES OF THE AUTHOR(S): Gaël Aglin <gael.aglin@uclouvain.be>"""
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

        best = LSNode(problem, max_state, step + 1)

    return best

# Attention : Depending of the objective function you use, your goal can be to maximize or to minimize it
def randomized_maxvalue(problem, limit=100, callback=None):
    current = LSNode(problem, problem.initial, 0)
    best = current

    for step in range(limit):
        if callback is not None:
            callback(current)

        successors = list(best.expand())
        successors.sort(key = lambda x: problem.fitness(x.state))

        rand = random.randint(0,4)
        best = LSNode(problem, successors[rand].state, step + 1)

    return best

#####################
#       Launch      #
#####################
if __name__ == '__main__':
    info = read_instance(sys.argv[1])
    init_state = State(info[0], info[1])
    bp_problem = BinPacking(init_state)

    step_limit = 100
    choose = 1

    if not choose:
        node = maxvalue(bp_problem, step_limit)
    else:
        node = randomized_maxvalue(bp_problem, step_limit)

    print(node.state)
