import numpy as np
import time

def open_test(filename):
    matrix = []
    file = open(filename, 'r')
    for val in file.read():
        if val == '.':
            matrix.append(0)
        else:
            matrix.append(int(val))
    matrix = np.array(matrix).reshape(9, 9)
    return matrix

def check(matrix, value, row, column):
    # check row
    if value in matrix[row, :]:
        return False
    # check column
    if value in matrix[:, column]:
        return False
    # check 3x3 box
    row = int(np.floor((row/3))*3)
    column = int(np.floor((column/3))*3)
    if value in matrix[row:row+3, column:column+3]:
        return False
    return True

class sudokuSolver:   
    def __init__(self, matrix):
        self._matrix = matrix

    def print_sudoku(self):
        for i in range(9):
            for j in range(9):
                print(self._matrix[i, j], end=' ')
                if j==2 or j==5:
                    print('|', end='')
            print('')
            if i==2 or i==5:
                print('------+------+-----')

    def solve(self, row=0, col=0):
        check_matrix = self._matrix.copy()
        if col == 9:
            if row == 8:
                self.print_sudoku()
                return True
            row += 1
            col = 0
        if self._matrix[row, col] == 0:
            for val in range(1, 10):
                self._matrix[row, col] = val
                if check(check_matrix, val, row, col):
                    self.solve(row, col+1)
            self._matrix[row, col] = 0
        else:
            return self.solve(row, col+1)
        return False

    # nieudana proba bez rekurencji
    # def solve(self):
    #     for i in range(9):
    #         for j in range(9):
    #             check_matrix = self._matrix.copy()
    #             if self._matrix[i, j] == 0:
    #                 self._matrix[i, j] = 1
    #                 if not check(check_matrix, self._matrix[i, j], i, j):                        
    #                     for x in range(1, 9):
    #                         self._matrix[i, j] += 1
    #                         if check(check_matrix, self._matrix[i, j], i, j):
    #                             break
    #                         if x == 8 and not check(check_matrix, self._matrix[i, j], i, j):
    #                             self._matrix[i, j] = 0
                                # self._matrix = check_matrix.copy()
        #         else:
        #             continue
        # self.print_sudoku()    


sudoku = open_test('test1.txt')
solver = sudokuSolver(sudoku)
solver.print_sudoku()
print('')
tic = time.perf_counter()
solver.solve()
toc = time.perf_counter()
print(f'Test case 1, czas: {toc-tic} sekund\n')


sudoku = open_test('test2.txt')
solver = sudokuSolver(sudoku)
solver.print_sudoku()
print('')
tic = time.perf_counter()
solver.solve()
toc = time.perf_counter()
print(f'Test case 2, czas: {toc-tic} sekund\n')


sudoku = open_test('test3.txt')
solver = sudokuSolver(sudoku)
solver.print_sudoku()
print('')
tic = time.perf_counter()
solver.solve()
toc = time.perf_counter()
print(f'Test case 3, czas: {toc-tic} sekund\n')