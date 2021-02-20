# -*-coding: utf-8 -*
'''NAMES OF THE AUTHOR(S): Gael Aglin <gael.aglin@uclouvain.be>'''
import time
import sys
from search import *


#################
# Problem class #
#################
class Knight(Problem):

    def successor(self, state):
        moves = [(-2, -1),(-1, -2),(-2, 1),(1, -2),(-1, 2),(2, -1),(1, 2),(2, 1)]
        old_pos = state.knight_pos
        states = []
        for i in range(8):
            new_pos = (old_pos[0] + moves[i][0],old_pos[1] + moves[i][1])
            if 0 <= new_pos[0] < state.nRows \
                and 0 <= new_pos[1] < state.nRows \
                and state.grid[new_pos[0]][new_pos[1]] == " ":

                potential = self.create_state(old_pos[0],old_pos[1],new_pos[0],new_pos[1],state)
                states.append(potential)

        for suc in states:
            yield (suc.knight_pos[0],suc.knight_pos[1]),suc

    def goal_test(self, state):
        goal = (state.nRows * state.nCols) + 1
        test = 0
        for i in range(state.nRows):
            for j in range(state.nCols):
                tmp = state.grid[i][j]
                if tmp == u"\u265E":
                    test += 1

                if tmp == u"\u2658":
                    test += 2

        return test == goal

    def create_state(self,old_x,old_y,new_x,new_y,state):
        new_state = State([state.nRows, state.nCols],(new_x, new_y))
        new_state.grid = [row[:] for row in state.grid]
        new_state.grid[old_x][old_y] = u"\u265E"
        new_state.grid[new_x][new_y] = u"\u2658"
        return new_state

###############
# State class #
###############

class State:
    def __init__(self, shapes, start_pos):
        self.nRows = shapes[0]
        self.nCols = shapes[1]
        self.grid = []
        for i in range(self.nRows):
            self.grid.append([" "] * self.nCols)

        self.knight_pos = start_pos
        self.grid[start_pos[0]][start_pos[1]] = "♘"

    def __str__(self):
        n_sharp = 2 * self.nCols + 1
        s = ("#" * n_sharp) + "\n"
        for i in range(self.nRows):
            s += "#"
            for j in range(self.nCols):
                s = s + str(self.grid[i][j]) + " "
            s = s[:-1]
            s += "#"
            if i < self.nRows - 1:
                s += '\n'
        s += "\n"
        s += "#" * n_sharp
        return s

    def tostring(self):
        for i in self.grid:
            print(i)
        print("\n")

##############################
# Launch the search in local #
##############################
#Use this block to test your code in local
# Comment it and uncomment the next one if you want to submit your code on INGInious

with open('instances.txt') as f:
    instances = f.read().splitlines()

for instance in instances:
    elts = instance.split(" ")
    shape = (int(elts[0]), int(elts[1]))
    init_pos = (int(elts[2]), int(elts[3]))
    init_state = State(shape, init_pos)

    problem = Knight(init_state)

    # example of bfs tree search
    startTime = time.perf_counter()
    node, nb_explored, remaining_nodes = breadth_first_graph_search(problem)
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
    else:
        print("bug lul")

'''
####################################
# Launch the search for INGInious  #
####################################
#Use this block to test your code on INGInious
shape = (int(sys.argv[1]),int(sys.argv[2]))
init_pos = (int(sys.argv[3]),int(sys.argv[4]))
init_state = State(shape, init_pos)

problem = Knight(init_state)

# example of bfs tree search
startTime = time.perf_counter()
node, nb_explored, remaining_nodes = breadth_first_graph_search(problem)
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