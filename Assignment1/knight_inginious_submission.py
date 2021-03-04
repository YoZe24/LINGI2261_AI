# -*-coding: utf-8 -*
# NAME OF THE AUTHOR : Gael Aglin <gael.aglin@uclouvain.be>
# Mehdi Ben Haddou <mehdi.benhaddou@student.uclouvain.be>
# Eliot Hennebo <eliot.hennebo@student.uclouvain.be>

from search import *

#################
# Problem class #
#################

# Legal moves of the knight
moves = [(-2, -1),(-1, -2),(-2, 1),(1, -2),(-1, 2),(2, -1),(1, 2),(2, 1)]

def create_state(old_x, old_y, new_x, new_y, state):
    """
    Takes the current state and modifies the position of the knight
    :param old_x: row of the current knight
    :param old_y: column of the current knight
    :param new_x: row of the generated knight position
    :param new_y: column of the generated knight position
    :param state: current state from which we generate the new position
    :return: new state containing the new position of the knight
    """
    new_state = State([state.nRows, state.nCols],(new_x, new_y))
    new_state.grid = [row[:] for row in state.grid]
    new_state.grid[old_x][old_y] = u"\u265E"
    new_state.grid[new_x][new_y] = u"\u2658"
    new_state.cost = state.cost + 1
    return new_state

def distance(state):
    """
    Compute the distance of the knight from the 2 closest borders of the board
    and assigns it directly in the object of the state
    :param state: State from which we want to compute the distance of the knight
    :return: void
    """
    x = state.knight_pos[0]
    y = state.knight_pos[1]
    state.dist = min(x,state.nRows - x - 1) + min(y,state.nCols - y - 1)

def check_legal_move(state,i):
    """
    Checks if a move is legal in a certain state
    :param state: State from which we want to check a move
    :param i: Move in the global array of moves
    :return: True if the move is legal
             False otherwise
    """
    old_pos = state.knight_pos
    new_pos = (old_pos[0] + moves[i][0],old_pos[1] + moves[i][1])
    return 0 <= new_pos[0] < state.nRows \
            and 0 <= new_pos[1] < state.nCols \
            and state.grid[new_pos[0]][new_pos[1]] == " "

class Knight(Problem):
    def successor(self, state):
        """
        Creates all the successors states from a certain state and returns them sorted according
        to our distance rule. It consists on computing the distance of the knight from the closest borders
        and then sorting the successors according their distances in increasing order
        :param state: State from which we want to calculate the successors
        :return: The list of valid successors sorted using our distance rule
        """
        old_pos = state.knight_pos
        states = []
        for i in range(8):
            new_pos = (old_pos[0] + moves[i][0], old_pos[1] + moves[i][1])
            if check_legal_move(state, i):
                potential = create_state(old_pos[0], old_pos[1], new_pos[0], new_pos[1], state)
                distance(potential)
                states.append(potential)

        states.sort(key = lambda x: x.dist, reverse=True)

        for suc in states:
            yield (suc.knight_pos[0], suc.knight_pos[1]), suc

    def goal_test(self, state):
        """
        Checks if a state is the goal state of the problem
        :param state: State that we want to check
        :return: True if the goal is reached
                 False otherwise
        """
        goal = (state.nRows * state.nCols)
        return state.cost + 1 == goal

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
        self.nbMoves = 0
        self.cost = 0
        self.dist = 0

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
node, nb_explored, remaining_nodes = depth_first_tree_search(problem)
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