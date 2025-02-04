"""
This module finds the solution of a given sudoku problem
Code credits: Tim Ruscica
More info: https://techwithtim.net/tutorials/python-programming/sudoku-solver-backtracking/
Example input board
board = [
    [7,8,0,4,0,0,1,2,0],
    [6,0,0,0,7,5,0,0,9],
    [0,0,0,6,0,1,0,7,8],
    [0,0,7,0,4,0,2,6,0],
    [0,0,1,0,5,0,9,3,0],
    [9,0,4,0,6,0,0,0,5],
    [0,7,0,3,0,0,0,1,2],
    [1,2,0,0,0,7,4,0,0],
    [0,4,9,2,0,6,0,0,7]
]
"""

def solve(bo):
    """
    Solves the given sudoku board using backtracking
    
    bo: 2D list representing the sudoku board
    """
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find
    for i in range(1,10):
        """
        Try each number from 1-9
        """
        if valid(bo, i, (row, col)):
            bo[row][col] = i
            if solve(bo):
                return True
            """
            If the current number doesn't work, reset the cell to 0
            """
            bo[row][col] = 0
    return False


def valid(bo, num, pos):
    """
    Checks if it is valid to place the given number at the given position

    bo: 2D list representing the sudoku board
    num: int representing the number to be placed
    pos: (row, col) representing the position to be checked

    Returns True if the number can be placed, False otherwise
    """
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False
    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False
    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bo[i][j] == num and (i,j) != pos:
                return False
    return True


def print_board(bo):
    """
    Prints the Sudoku board in a formatted manner with grid lines.

    Parameters:
    bo (list): A 2D list representing the Sudoku board.
    """
    for i in range(len(bo)):
        # Print horizontal separator every 3 rows
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")
        
        for j in range(len(bo[0])):
            # Print vertical separator every 3 columns
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            
            # Move to the next line after the last column
            if j == 8:
                print(bo[i][j])
            else:
                # Print each number followed by a space
                print(str(bo[i][j]) + " ", end="")


def find_empty(bo):
    """
    Finds the first empty cell in the Sudoku board.

    Parameters:
    bo (list): A 2D list representing the Sudoku board.

    Returns:
    tuple: A tuple of (row, col) representing the position of the empty cell if found, None otherwise.
    """
    # Iterate over each cell in the Sudoku board
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            # Check if the current cell is empty
            if bo[i][j] == 0:
                # Return the position of the first empty cell
                return (i, j)  # row, col
    # Return None if no empty cell is found
    return None


# test code
# solve(board)
# print_board(board)