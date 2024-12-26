import numpy as np
from copy import deepcopy

rotationMatrix_90degrees = [
    [1,0,0,0],
    [0,-0,-1,0],
    [0,1,-0,0],	
    [0,0,0,1]]

class Game:

    def __init__(self, inputs = None):
        
        if (inputs != None):
            self.board = []
            for r in range(4):
                l = []
                for c in range(4):
                    l.append(inputs[(4*r)+c])
                    
                self.board.append(l)
        else:
            self.board = [
                [0,0,0,0],
                [0,0,0,0],
                [0,0,0,0],
                [0,0,0,0]
            ]
        self.board = np.array(self.board)
    
    def inputs(self):
        i = []
        for y in self.board:
            for x in y:
                i.append(x)
        
        return i
    
    def legal_placements(self):
        placements = []
        for r in range(4):
            for c in range(4):
                if (self.board[r][c] == 0):
                    placements.append(1)
                else:
                    placements.append(0)
        return placements

    def has_won(self):
        for i in range(2):
            for row in self.board:
                testSum = 0
                for i,val in enumerate(row):
                    if (val == 0 and i != 0): break
                    testSum += val
                if (testSum >= 3): return 1
                if (testSum <= -3): return -1
            board = board.T
        
        # for r in range(len(self.board)):
        #     testSum = 0
        #     for c in range(len(self.board[r])):
        #         val = self.board[c][r]
        #         if (val == 0 and i != 0): break
        #         testSum += val
        #     if (testSum >= 3): return 1
        #     if (testSum <= -3): return -1

        diagonals = [0,0,0,0,0,0]

        for i in range(3):
            diagonals[0] += self.board[1+i][i]
            diagonals[1] += self.board[i][1+i]
            diagonals[2] += self.board[2-i][i]
            diagonals[3] += self.board[3-i][1+i]

        for i in range(4):
            diagonals[4] += self.board[i][i]
            diagonals[5] += self.board[3-i][i]
        
        for val in diagonals: 
            if (val >= 3): return 1
            if (val <= -3): return -1
        
        return 0

    def test_player1_place(self, position):
        #if (position[1] < 0 and position[0] < 0): return False
        if (self.board[position[1]][position[0]] == 0):
            return True
        else:
            return False
    
    def test_player2_place(self, position):
        #if (position[1] < 0 and position[0] < 0): return False
        if (self.board[position[1]][position[0]] == 0):
            return True
        else:
            return False
