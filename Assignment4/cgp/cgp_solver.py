from clause import *

from Assignment4.cgp.clause import Clause

"""
For the color grid problem, the only code you have to do is in this file.

You should replace

# your code here

by a code generating a list of clauses modeling the grid color problem
for the input file.

You should build clauses using the Clause class defined in clause.py

Read the comment on top of clause.py to see how this works.
"""


def get_moves():
    pass


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
                for wtf in range(size):
                    get_moves()

            expression.append(cell_color_clause)

    return expression


if __name__ == '__main__':
    expression = get_expression(3)
    for clause in expression:
        print(clause)
