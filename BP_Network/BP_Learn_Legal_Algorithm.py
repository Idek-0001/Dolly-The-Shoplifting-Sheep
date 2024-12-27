from BP_Neural_Network import Network
from Training_Board import Game
import numpy as np
import time


# class Algorithm:

#     def __init__(self, population, PREFERENCES):
#         self.PREFERENCES = PREFERENCES
#         self.population = population
#         self.population_size = len(population)

    
#     #def get_child(self, network1, network2)

#     def train_generation(self):
#         pass

def get_board_placement_index(network):
    index = 0
    prev = 0
    for i, val in enumerate(network.outputs):
        if (val > prev):
            index = i
            prev = val

    return index

def train_legal_placements():

    n = Network([16,16,16], learning_rate= 0.00001)



    boards = [Game([1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]) for b in range(1)]

    #print(str(boards[0].legal_placements()))

    tic = time.process_time()
    for board in boards:
        legal_placements = np.array(board.legal_placements())
        while (np.mean(np.square(legal_placements - n.outputs)) > 0.2):
            n.propogate(board.inputs())
            n.backpropogate(legal_placements - n.outputs, n.hidden_layers)

            print(np.mean(np.square(legal_placements - n.outputs)))
        print(n.weights)



    toc = time.process_time()
    print("Computation time = " + str(1000*(toc - tic )) + "ms")  


def get_op_learningRate():
    board = Game(inputs = [1,1,-1,0,1,1,-1,-1,1,-1,1,-1,1,-1,1,-1])
    inputs = board.inputs()
    legal_placements = np.array(board.legal_placements())
    
    print(legal_placements)

    iter_nums = []

    for LR in range(1):
        LR += 1
        n = Network([16,13,16], learning_rate= 2.0)
        average = 0
        for a in range(4):
            i = 0
            while (legal_placements[get_board_placement_index(n)] != 1 and i < 100000):
                n.propogate(inputs)
                n.backpropogate(legal_placements - n.outputs, n.hidden_layers)
                average += 1
                i += 1
                #print(i)
                #

                if (i % 10000 == 0): 
                    print(get_board_placement_index(n))
                    print(np.mean(np.square(legal_placements - n.outputs)))
        #print("RUNNING")
        iter_nums.append( [(LR) , average/4] )

    #[print(i%0.1) for i[1] in iter_nums]
    #[print(i) for i in n.outputs]
    print(n.outputs)

    print(iter_nums.sort(reverse=True, key=lambda item :item[1]))


def test_learning():
    n = Network([16,13,16], learning_rate=0.1)
    inputs = [0,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1]
    legal_placements = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    i = 0
    while np.mean(np.square(legal_placements - n.outputs)) > 0.01 and i < 100000:
        n.propogate(inputs)
        n.backpropogate(legal_placements - n.outputs, n.hidden_layers)

        if (i % 1000 == 0):
            #print( str(i) + ", ERROR: " + str(np.mean(np.abs(legal_placements - n.outputs))))
            #print(legal_placements - n.outputs)
            #print(np.where(n.outputs == max(n.outputs)))
            print(n.outputs)
            #print(str(n.outputs[0]) + "  " +  str(n.outputs[1]))
            #print(n.weights[1][1])
            #print(i)
        i += 1
        #n.learning_rate = np.sqrt(i)

    


test_learning()
#train_legal_placements()
#get_op_learningRate()       
 
