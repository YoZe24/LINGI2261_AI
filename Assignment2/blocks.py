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
        successors = []
        for block in state.blocks_positions:
            old_x,old_y = block
            val = state.blocks_positions[block]
            if val != '@' and val.islower():
                left = (old_x, old_y - 1)
                right = (old_x, old_y + 1)
                if left[1] >= 0 and (state.grid[left[0]][left[1]] == ' ' or grid_goal[left[0]][left[1]].isupper()):
                    grid_left = []
                    for row in state.grid:
                        grid_left.append(row[:])

                    blocks_positions_left = dict(state.blocks_positions)
                    suc_left = State(grid_left,state.blocks_remaining,blocks_positions_left,state.goals)
                    if grid_goal[left[0]][left[1]].isupper():
                        if grid_goal[left[0]][left[1]] == suc_left.grid[old_x][old_y].upper():
                            suc_left.grid[left[0]][left[1]] = '@'
                            suc_left.blocks_remaining -= 1
                    else:
                        suc_left.grid[left[0]][left[1]] = val
                    suc_left.grid[old_x][old_y] = ' '
                    suc_left.compute_blocks_positions()
                    suc_left.compute_gravity()
                    successors.append(("l", suc_left))

                if right[1] < len(state.grid[0]) and (state.grid[right[0]][right[1]] == ' ' or grid_goal[right[0]][right[1]].isupper()):
                    grid_right = []
                    for row in state.grid:
                        grid_right.append(row[:])

                    blocks_positions_right = dict(state.blocks_positions)
                    suc_right = State(grid_right,state.blocks_remaining,blocks_positions_right,state.goals)
                    if grid_goal[right[0]][right[1]].isupper():
                        if grid_goal[right[0]][right[1]] == suc_right.grid[old_x][old_y].upper():
                            suc_right.grid[right[0]][right[1]] = '@'
                            suc_right.blocks_remaining -= 1
                    else:
                        suc_right.grid[right[0]][right[1]] = val
                    suc_right.grid[old_x][old_y] = ' '
                    suc_right.compute_blocks_positions()
                    suc_right.compute_gravity()
                    successors.append(("l", suc_right))

        return successors

    def goal_test(self, state):
        return state.blocks_remaining == 0

###############
# State class #
###############
class State:
    def __init__(self, grid, blocks_remaining = -1, blocks_positions = None, goals = None):
        self.nbr = len(grid)
        self.nbc = len(grid[0])
        self.grid = grid

        if blocks_remaining == -1 and blocks_positions is None:
            self.blocks_remaining = 0
            self.compute_blocks_remaining()
            self.blocks_positions = {}
            self.compute_blocks_positions()
            self.goals = {}
            self.compute_goals()
            self.compute_gravity()
        else:
            self.blocks_remaining = blocks_remaining
            self.blocks_positions = blocks_positions
            self.goals = goals

    def compute_blocks_remaining(self):
        count_done = 0
        count_todo = 0
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                done = self.grid[i][j]
                todo = grid_goal[i][j]
                if done == '@':
                    count_done += 1
                if todo != ' ' and todo != '#' and todo.isupper():
                    count_todo += 1
        self.blocks_remaining = count_todo - count_done

    def compute_blocks_positions(self):
        self.blocks_positions = {}
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                cell = self.grid[i][j]
                goal_cell = grid_goal[i][j]
                if cell != ' ' and cell != '#' and cell.islower:
                    self.blocks_positions[(i,j)] = cell
                if goal_cell != ' ' and goal_cell != '#' and goal_cell.isupper:
                    if cell == '@':
                        self.blocks_positions[(i,j)] = goal_cell
        self.sort_blocks_positions()

    def sort_blocks_positions(self):
        self.blocks_positions = {key: self.blocks_positions[key] for key in sorted(self.blocks_positions,key = lambda el : el[0], reverse= True)}

    def merge_grids(self):
        for i in range(len(grid_goal)):
            for j in range(len(grid_goal[0])):
                cell = grid_goal[i][j]
                if cell != ' ' and cell != '#' and cell.isupper:
                    self.grid[i][j] = cell

    def compute_gravity(self):
        for block in self.blocks_positions:
            val = self.blocks_positions[block]
            if val.islower():
                x,y = block
                while x + 1 < self.nbr and self.grid[x + 1][y] != '@' and ((y < self.nbc and self.grid[x + 1][y] != '#' and self.grid[x + 1][y] == ' ') or grid_goal[x + 1][y].isupper()):
                    if grid_goal[x + 1][y].isupper():
                        if self.grid[x][y].upper() == grid_goal[x + 1][y]:
                            self.grid[x + 1][y] = '@'
                            self.blocks_remaining -= 1
                        else:
                            self.grid[x + 1][y] = self.grid[x][y]
                    else:
                        self.grid[x + 1][y] = self.grid[x][y]

                    self.grid[x][y] = ' '
                    x = x + 1
                    self.compute_blocks_positions()

    def get_heuristic(self):
        sum_dist = 0
        for position, block in self.goals.items():
            x, y = position
            if block.isupper():
                dist = self.distance(x, y, block)
                sum_dist += dist

        return sum_dist

    def distance(self, x_goal, y_goal, target):
        min_dist = -1
        for (x, y), block in self.blocks_positions.items():
            if block.islower() \
                    and block.upper() == target \
                    and x <= x_goal \
                    and ((min_dist == -1) or (abs(y - y_goal) < min_dist)):
                min_dist = abs(y - y_goal)

        return int(min_dist)

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

    def compute_goals(self):
        self.goals = {}
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                goal_cell = grid_goal[i][j]
                if goal_cell != ' ' and goal_cell != '#' and goal_cell.isupper:
                    self.goals[(i, j)] = goal_cell


######################
# Auxiliary function #
######################
def readInstanceFile(filename):
    grid_init, grid_goal = map(lambda x: [[c for c in l.rstrip('\n')[1:-1]] for l in open(filename + x)], [".init", ".goalinfo"])
    return grid_init[1:-1], grid_goal[1:-1]

######################
# Heuristic function #
######################
def heuristic(heu):
    return heu.state.get_heuristic()

##############################
# Launch the search in local #
##############################
#Use this block to test your code in local
# Comment it and uncomment the next one if you want to submit your code on INGInious
instances_path = "instances/"
instance_names = ['a01','a02','a03','a04','a05','a06','a07','a08','a09','a10']

#for instance in [instances_path + name for name in instance_names]:
grid_init, grid_goal = readInstanceFile(instances_path + instance_names[0])
init_state = State(grid_init)
# goal_state = State(grid_goal)
problem = Blocks(init_state)

# example of bfs tree search
startTime = time.perf_counter()
node, nb_explored, remaining_nodes = astar_graph_search(problem,heuristic)
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