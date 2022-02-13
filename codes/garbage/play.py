import chess
from MCTS_nn import *
from nn import *
from IPython.display import display

class game():


    def __init__(self):

        self.board = chess.Board()


    def play_random(self):

        random_move = sample(list(self.board.legal_moves),1)[0]
        self.board.push(random_move)

        display(self.board)

        return


    def play_mcts_nn(self, nb_simul):

        MCTS = mcts_nn(self.board)

        for i in range(nb_simul):
            MCTS.expansion_backprop(MCTS.selection())

        N = [child.N for child in MCTS.root.children]
        index = sample([i for i, j in enumerate(N) if j == max(N)],1)[0] 
        move = [child.move for child in MCTS.root.children][index]
        self.board.push(move)

        display(self.board)

        return 
