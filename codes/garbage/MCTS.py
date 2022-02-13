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
        return relative_W/(self.N or 1) + np.sqrt(2)*np.sqrt(np.log((self.parent.N or 1))/(self.N or 1))


class mcts():
    def __init__(self,position):
        self.initial_position = position.copy() # initial_position and current_position are chess.Board() objects 
        self.current_position = position.copy()
        self.root = node() # create the tree root, that correspond to initial_position

    def selection(self): # we reach a leaf using UCT selection
        current_node = self.root # we start from the root
        while current_node.is_leaf is not True: # check if we are in a leaf
            white_to_play = self.current_position.turn
            UCTs = [child.uct(white_to_play) for child in current_node.children] # list of UCTs for the node's children
            index = sample([i for i, j in enumerate(UCTs) if j == max(UCTs)],1)[0] # si plusieurs max tirage au hasard
            current_node = current_node.children[index] # the best children (with maximal UCT) becomes the current node
            self.current_position.push(current_node.move) # mise à jour de la position courante pour la phase d'expansion / simulation
        leaf = current_node # the current node is a leaf by definition
        return leaf

    def expansion_simulation(self,leaf): # we reach a final position starting from a previous leaf using random moves

        outcome = self.current_position.outcome() # résultat actuel de la partie
        #EXPANSION
        legal_moves = self.current_position.legal_moves # génération des coups légaux
        child_node = None # juste pour ne pas bugger la fin de partie

        if outcome is None: # si la partie n'est pas terminée
            chosen_move = sample(list(legal_moves),1)[0] # tirage aléatoire d'un coup légal
            for move in legal_moves: # création des nouveaux noeuds correspondants aux coups légaux
                if move == chosen_move:
                    child_node = node(move=move,parent=leaf) # On garde le noeud enfant dans une variable, on en aura besoin dans la rétropropagation
                else:
                    node(move=move,parent=leaf)
            self.current_position.push(chosen_move) # mise à jour de la position courante avec le nouveau coup
            outcome = self.current_position.outcome() # mise à jour de l'issue de la partie
        #SIMULATION
        while outcome is None : # tant que la partie n'est pas terminée
            legal_moves = self.current_position.legal_moves # génération des coups légaux
            random_move = sample(list(legal_moves),1)[0] # tirage aléatoire du prochain coup
            self.current_position.push(random_move) # mise à jour de la position courante avec le nouveau coup
            outcome = self.current_position.outcome() # mise à jour de l'issue de la partie
        result = outcome.result() 

        return child_node, result

    def backpropagation(self, child_node, result): # we backpropagate information through the tree
        if result == "1-0":
            inc = 1
        elif result == "0-1":
            inc = -1
        else:
            inc = 0
        if child_node is not None :  
            for ancestor in child_node.iter_path_reverse(): # rétropropagation du résultat issu de la simulation
                ancestor.N += 1
                ancestor.W += inc
        self.current_position = self.initial_position.copy()
        return