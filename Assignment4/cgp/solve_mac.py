#!/usr/bin/env python3
import sys
from cgp_solver import get_expression
import minisat


def default_usage():
    # The argument must reference an instance file and the second
    print("Usage:", sys.argv[0], "INSTANCE_FILE", file=sys.stderr)
    exit(1)


def get_row(grid, i):
    return grid[i]


def get_column(grid, j):
    out = []
    for i in range(len(grid)):
        out.append(grid[i][j])
    return out


def get_left_diag(grid, i, j):
    out = []
    for add in range(-len(grid), len(grid)):
        if i+add >= 0 and i + add < len(grid) and j+add >= 0 and j+add < len(grid):
            out.append(grid[i+add][j+add])
    return out


def get_right_diag(grid, i, j):
    out = []
    for add in range(-len(grid), len(grid)):
        if i + add >= 0 and i + add < len(grid) and j - add >= 0 and j - add < len(grid):
            out.append(grid[i + add][j - add])
    return out


def is_all_diff(colors):
    return len(set(colors)) == len(colors)


def read_instance(instance_file):
    file = open(instance_file)
    size = int(file.readline().split(' ')[0])
    points = []
    line = file.readline()
    while line:
        points.append((int(line.split(' ')[0]), int(line.split(' ')[1]), int(line.split(' ')[2])))
        line = file.readline()
    return size, points


if __name__ == "__main__":
    if len(sys.argv) != 2:
        default_usage()

    size, points = read_instance(sys.argv[1])
    n_rows = n_columns = n_colors = size
    expression = get_expression(size, points)
    nb_vars = n_rows * n_columns * n_colors
    solution = minisat.minisat(nb_vars, [clause.minisat_str() for clause in expression], './minisatMac')

    if solution is None:
        print("The problem is unfeasible")
        exit(0)
    grid = []
    i = 1
    tmp = []
    for s in solution:
        if s % size == 0:
            tmp.append(size-1)
        else:
         tmp.append(s % size - 1)
        i += 1
        if len(tmp) == size:
            grid.append(tmp)
            i = 1
            tmp = []

    clean = True
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if not is_all_diff(get_row(grid, i)):
                clean = False
                print("FAIL. Row {0} is not different.".format(i))
            if not is_all_diff(get_column(grid, j)):
                clean = False
                print("FAIL. Column {0} is not different.".format(i))
            if not is_all_diff(get_left_diag(grid, i, j)):
                clean = False
                print("FAIL. Left diagonal passing through cell ({0},{1}) is not different.".format(i, j))
            if not is_all_diff(get_right_diag(grid, i, j)):
                clean = False
                print("FAIL. Right diagonal passing through cell ({0},{1}) is not different.".format(i, j))
    for point in points:
        if grid[point[0]][point[1]] != point[2]:
            clean = False
            print("FAIL. Point ({0},{1}) is not equal to {2} as required".format(point[0], point[1], point[2]))
    if clean:
        print("SOLVED")
        for row in grid:
            print(row)
