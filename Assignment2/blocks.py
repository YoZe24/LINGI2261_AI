# -*-coding: utf-8 -*
"""
NAME OF THE AUTHOR :
- Gaël Aglin <gael.aglin@uclouvain.be>
NAMES OF THE STUDENTS :
- Mehdi Ben Haddou <mehdi.benhaddou@uclouvain.be>
- Eliot Hennebo <eliot.hennebo@uclouvain.be>
"""
from search import *
import sys
import time

goal_state = None
#################
# Problem class #
#################
class Blocks(Problem):

    def successor(self, state):
        return []

    def goal_test(self, state):
        return state.blocks_remaining == 0


###############
# State class #
###############
class State:
    def __init__(self, grid, blocks_remaining = -1, blocks_positions = None):
        self.nbr = len(grid)
        self.nbc = len(grid[0])
        self.grid = grid

        if blocks_remaining == -1 and blocks_positions is None:
            self.merge_grids()
            self.blocks_remaining = 0
            self.compute_blocks_remaining()
            self.blocks_positions = {}
            self.compute_blocks_positions()
        else:
            self.blocks_remaining = blocks_remaining
            self.blocks_positions = blocks_positions


        self.compute_gravity()
        print(self.blocks_positions)
        print(self.blocks_remaining)
        print(self)

    def compute_blocks_remaining(self):
        count = 0
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                cell = self.grid[i][j]
                if cell != ' ' and cell != '#' and self.grid[i][j].isupper():
                    count += 1

        self.blocks_remaining = count

    def compute_blocks_positions(self):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                cell = self.grid[i][j]
                if cell != ' ' and cell != '#' and cell.islower:
                    self.blocks_positions[(i,j)] = cell

    def merge_grids(self):
        for i in range(len(grid_goal)):
            for j in range(len(grid_goal[0])):
                cell = grid_goal[i][j]
                if cell != ' ' and cell != '#' and cell.isupper:
                    self.grid[i][j] = cell

    def compute_gravity(self):
        # TODO : trier blocks_positions en fonction des X croissants ainsi on commence par les blocks les plus en bas pour la gravité
        for block in self.blocks_positions:
            if self.blocks_positions[block].islower():
                x,y = block
                while x + 1 < self.nbr and y < self.nbc and self.grid[x + 1][y] != '#' and self.grid[x + 1][y] == ' ' and not(self.grid[x + 1][y].isupper()):
                    self.grid[x + 1][y] = self.grid[x][y]
                    self.grid[x][y] = ' '
                    x += 1

    def __str__(self):
        n_sharp = self.nbc + 2
        s = ("#" * n_sharp) + "\n"
        for i in range(self.nbr):
            s += "#"
            for j in range(self.nbc):
                s = s + str(self.grid[i][j])
            s += "#"
            if i < self.nbr - 1:
                s += '\n'
        return s + "\n" + "#" * n_sharp


class Block:
    def __init__(self,x,y,goal_x,goal_y):
        self.x = x
        self.y = y
        self.goal_x = goal_x
        self.goal_y = goal_y

######################
# Auxiliary function #
######################
def readInstanceFile(filename):
    grid_init, grid_goal = map(lambda x: [[c for c in l.rstrip('\n')[1:-1]] for l in open(filename + x)], [".init", ".goalinfo"])
    return grid_init[1:-1], grid_goal[1:-1]

######################
# Heuristic function #
######################
def heuristic(node):
    h = 0.0
    # ...
    # compute an heuristic value
    # ...
    return h

##############################
# Launch the search in local #
##############################
#Use this block to test your code in local
# Comment it and uncomment the next one if you want to submit your code on INGInious
instances_path = "instances/"
instance_names = ['a01','a02','a03','a04','a05','a06','a07','a08','a09','a10']

n = 1
for instance in [instances_path + name for name in instance_names]:
    print("Instance : ", n)
    grid_init, grid_goal = readInstanceFile(instance)
    init_state = State(grid_init)
    # goal_state = State(grid_goal)
    problem = Blocks(init_state)

    # example of bfs tree search
    startTime = time.perf_counter()
    node, nb_explored, remaining_nodes = depth_first_graph_search(problem)
    endTime = time.perf_counter()

    # example of print

    if node:
        path = node.path()
        path.reverse()

        print('Number of moves: ' + str(node.depth))
        for n in path:
            print(n.state)  # assuming that the __str__ function of state outputs the correct format
            print()
        print("* Execution time:\t", str(endTime - startTime))
        print("* Path cost to goal:\t", node.depth, "moves")
        print("* #Nodes explored:\t", nb_explored)
        print("* Queue size at goal:\t", remaining_nodes)

    n += 1
    print("-----------------------------------------")

####################################
# Launch the search for INGInious  #
####################################
#Use this block to test your code on INGInious
'''
instance = sys.argv[1]
grid_init, grid_goal = readInstanceFile(instance)
init_state = State(grid_init)
goal_state = State(grid_goal)
problem = Blocks(init_state)

# example of bfs graph search
startTime = time.perf_counter()
node, nb_explored, remaining_nodes = astar_graph_search(problem, heuristic)
endTime = time.perf_counter()

# example of print
path = node.path()
path.reverse()

print('Number of moves: ' + str(node.depth))
for n in path:
    print(n.state)  # assuming that the __str__ function of state outputs the correct format
    print()
print("* Execution time:\t", str(endTime - startTime))
print("* Path cost to goal:\t", node.depth, "moves")
print("* #Nodes explored:\t", nb_explored)
print("* Queue size at goal:\t",  remaining_nodes)
'''