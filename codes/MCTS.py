from anytree import NodeMixin
import numpy as np
import chess
from random import sample

class node(NodeMixin):

    def __init__(self,move=None,parent=None):
        self.move = move 
        self.N = 0
        self.W = 0
        self.parent = parent

    def uct(self,white_to_play): # compute uct 
        if white_to_play:
            relative_W = self.W
        else:
            relative_W = -self.W
        return relative_W/(self.N+1e-5) + np.sqrt(2)*np.sqrt(self.parent.N)/(self.N+1e-5)


class mcts():

    def __init__(self,position):
        self.initial_position = position # initial_position and current_position are chess.Board() objects 
        self.current_position = position
        self.tree = node() # create the tree root, that correspond to initial_position

    def selection(self): # we reach a leaf using UCT selection
        current_node = self.tree # we start from the root
        while current_node.is_leaf is not True: # check if we are in a leaf
            white_to_play = self.current_position.turn
            UCTs = [child.uct(white_to_play) for child in current_node.children] # list of UCTs for the node's children
            current_node = current_node.children[np.argmax(UCTs)] # the best children (with maximal UCT) becomes the current node
            self.current_position.push(current_node.move) # mise à jour de la position courante pour la phase d'expansion / simulation
        leaf = current_node # the current node is a leaf by definition
        return leaf

    def expansion_simulation(self,leaf): # we reach a final position starting from a previous leaf using random moves
        current_node = leaf # the current node is a leaf (termination) by definition
        result = self.current_position.outcome().result() # résultat actuel de la partie
        while result is None : # tant que la partie n'est pas terminée
            legal_moves = self.current_position.legal_moves # génération des coups légaux
            random_move = sample(legal_moves) # tirage aléatoire du prochain coup
            # création du nouveau noeud correspondant (penser à mettre à jour parents et enfants, et le coup associé)
            new_node = node(move=random_move,parent=current_node)
            current_node = new_node
            # mise à jour de la position courante avec le nouveau coup
            self.current_position.push(random_move)
            # mettre à jour la variable result
            result = self.current_position.outcome().result()
        termination_node = current_node # le noeud courant correspond à une fin de partie par définition 
        return termination_node, result


    def backpropagation(self, termination_node, result): # we backpropagate information through the tree
        if result == "1-0":
            inc = 1
        elif result == "0-1":
            inc = -1
        else:
            inc = 0
        for ancestor in termination_node.ancestors: # rétropropagation du résultat issu de la simulation
            ancestor.N += 1
            ancestor.W += inc
        self.current_position = self.initial_position
        return

