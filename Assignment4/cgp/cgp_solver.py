from clause import *

# from Assignment4.cgp.clause import Clause

"""
For the color grid problem, the only code you have to do is in this file.

You should replace

# your code here

by a code generating a list of clauses modeling the grid color problem
for the input file.

You should build clauses using the Clause class defined in clause.py

Read the comment on top of clause.py to see how this works.
"""


def get_moves(dist):
    return [(0,-dist), (-dist, 0),  (-dist,dist),  (-dist,-dist)]
    # return [(0,dist), (dist,0), (dist,-dist), (dist,dist)]


def get_expression(size, points=None):
    expression = []

    if points is not None:
        for i,j,k in points:
            clause = Clause(size)
            clause.add_positive(i,j,k)
            expression.append(clause)

    for i in range(size):
        for j in range(size):
            cell_color_clause = Clause(size)
            for k in range(size):
                cell_color_clause.add_positive(i,j,k)
                for dist in range(1,i+1):

                    for dist_i, dist_j in get_moves(dist):
                        x,y = i+dist_i, j+dist_j

                        if x != i or y != j:
                            if 0 <= x < size and 0 <= y < size:
                                star_clause = Clause(size)
                                star_clause.add_negative(i,j,k)
                                star_clause.add_negative(x,y,k)
                                expression.append(star_clause)

            expression.append(cell_color_clause)

    return expression


if __name__ == '__main__':
    expression = get_expression(3)
    for clause in expression:
        print(clause)
