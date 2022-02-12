from anytree import NodeMixin
import numpy as np
import chess
from random import sample
from utils import *
from nn import *


class node_nn(NodeMixin):

    def __init__(self,move=None,parent=None,prob=0,V=0):
        self.move = move 
        self.N = 0
        self.V = V
        self.prob = prob
        self.parent = parent

    def score(self,white_to_play):
        if white_to_play:
            relative_V = self.V
        else:
            relative_V = -self.V
        return relative_V/(self.N or 1) + (1/(self.N+1))*self.prob * np.sqrt(self.parent.N) / (1 + self.N)
        

class mcts_nn():
    def __init__(self,position):
        self.initial_position = position.copy() # initial_position and current_position are chess.Board() objects 
        self.current_position = position.copy()
        self.root = node_nn() # create the tree root, that correspond to initial_position
        self.model = load_model() # load nn model from nn.py
        self.moves = [chess.Move.from_uci(move) for move in create_uci_labels()]

    def selection(self): # we reach a leaf using score selection
        current_node = self.root # we start from the root
        while current_node.is_leaf is not True: # check if we are in a leaf
            white_to_play = self.current_position.turn
            score = [child.score(white_to_play) for child in current_node.children] # list of scores for the node's children
            index = sample([i for i, j in enumerate(score) if j == max(score)],1)[0] # si plusieurs max tirage au hasard
            current_node = current_node.children[index] # the best children (with maximal UCT) becomes the current node
            self.current_position.push(current_node.move) # mise à jour de la position courante pour la phase d'expansion / simulation
        leaf = current_node # the current node is a leaf by definition
        return leaf

    def expansion(self,leaf): 
        outcome = self.current_position.outcome() 
        #EXPANSION
        legal_moves = self.current_position.legal_moves # génération des coups légaux
        if outcome is None: # si la partie n'est pas terminée
            p,v = evaluate_position(self.model, self.current_position)
            p = p[0] # juste pour des questions de dimensions
            v = v[0,0]
            leaf.V += v
            for move in legal_moves: # création des nouveaux noeuds correspondants aux coups légaux
                prob = p[self.moves.index(move)]
                node_nn(move=move,parent=leaf,prob=prob)
        return leaf, v

    def backpropagation(self, leaf, v): # we backpropagate information through the tree
  
        for ancestor in leaf.iter_path_reverse(): # rétropropagation du résultat issu de la simulation
            ancestor.N += 1
            ancestor.V += v

        self.current_position = self.initial_position.copy()

        return