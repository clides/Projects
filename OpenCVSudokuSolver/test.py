import sudokuSolver

board = [[0, 7, 0, 0, 2, 0, 0, 4, 6], [0, 6, 0, 0, 0, 0, 8, 9, 0], [2, 0, 0, 8, 0, 0, 7, 1, 5], [0, 8, 4, 0, 9, 7, 0, 0, 0], [7, 1, 0, 0, 0, 0, 0, 5, 9], [0, 0, 0, 1, 3, 0, 4, 8, 0], [6, 9, 7, 0, 0, 2, 0, 0, 8], [0, 5, 8, 0, 0, 0, 0, 6, 0], [4, 3, 0, 0, 8, 0, 0, 7, 0]]

# Try solving the board
try:
    # Call the solve function from sudoku_solver and pass the board
    solved = sudokuSolver.solve(board)
    
    # Check if a solution was found
    if solved:
        print("\nSolved board:")
        print(board)
    else:
        print("\nNo solution found.")
        
except Exception as e:
    print(f"An error occurred: {e}")