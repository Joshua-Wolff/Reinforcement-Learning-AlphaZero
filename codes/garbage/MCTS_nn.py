from anytree import NodeMixin # librairie pour les structures d'arbre
import numpy as np 
import chess # librairie d'échec, affichage, règles, coups légaux...
from random import sample # tirage aléatoire
from utils import * # fonctions utilitaires, cf 
from nn import * # appel au réseau de neurones pré-entraîné pour l'évaluation des positions
from numpy.random import dirichlet # tirage selon une loi de Dirichlet


'''
MCTS_nn.py

Contient les classes node_nn et mcts_nn qui permettent d'implémenter 
une recherche arborescente Monte Carlo améliorée par un réseau de neurone 
pré-entraîné qui fournit une politique d'expert.
'''

class node_nn(NodeMixin):

    '''
    Classe node_nn

    Cette classe correspond aux noeuds de l'arbre. Elle est caractérisée par 5 attributs :

        - move : chess.Move, coup correspondant au noeud
        - N : entier, nombre de fois où le noeud a été visité
        - V : réel, somme des valuations de la position
        - prob : réel, probabilité de choisir le coup correspondant au noeud sachant qu'on était dans la position précédente
        - parent : le noeud correspondant à la position précédente
    '''

    def __init__(self,move=None,parent=None,prob=0,V=0):

        self.move = move 
        self.N = 0
        self.V = V
        self.prob = prob
        self.parent = parent

    '''
    Fonction score(node_nn(), white_to_play)

    Argument :
        - white_to_play : booléen, vrai si c'est aux blancs de jouer

    Sortie :
        - Score qui permet de choisir quel noeud choisir lors de la phase de sélection de l'algorithme MCTS
    '''

    def score(self,white_to_play):

        if white_to_play:
            relative_V = self.V

        else:
            relative_V = -self.V

        return relative_V/(self.N or 1) + 1.5 * self.prob * np.sqrt(self.parent.N) / (1 + self.N)
        




class mcts_nn():

    '''
    Classe mcts_nn

    Cette classe correspond à l'arbre dans lequel on effectue MCTS. Elle est caractérisée par 6 attributs :

        - initial_position : chess.Board(), position dans laquelle on est réellement
        - current_position : chess.Board(), variable utilisé pour stocker les différentes positions courantes rencontrées dans MCTS
        - root : node_nn(), noeud correspondant à la position initial_position
        - model : modèle keras, le réseau de neurone utilisé pour cacluler les valuations des positions et les probabilités conditionnelles
        - moves_w : tous les coups jouables pour les blancs
        - moves_b : tous les coups jouables pour les noirs
    '''

    def __init__(self,position):

        self.initial_position = position.copy() 
        self.current_position = position.copy()
        self.root = node_nn()
        self.model = load_model()
        self.moves_w = [chess.Move.from_uci(move) for move in create_uci_labels()]
        self.moves_b = [chess.Move.from_uci(move) for move in flipped_uci_labels()]

    '''
    Fonction selection(mcts_nn())

    Arguments :

    Sorties :
        - un noeud de l'arbre qui est une feuille

    Description : 
        Cette fonction permet, à partir de la racine de l'arbre, de sélectionner une feuille en parcourant les noeuds 
        ayant les scores les plus hauts. La variable current_position est parallèlement mise à jour en fonction des noeuds empruntés.
        On retourne finalement le noeud dans lequel on aboutit (c'est nécessairement une feuille).
    '''
    
    def selection(self): 

        current_node = self.root

        while current_node.is_leaf is not True: # tant qu'on est pas arrivé dans une feuille

            white_to_play = self.current_position.turn
            score = [child.score(white_to_play) for child in current_node.children] # calcul des scores des noeuds enfants du noeud courant
            index = sample([i for i, j in enumerate(score) if j == max(score)],1)[0] # si plusieurs scores sont maximaux on en tire un au hasard parmi ces noeuds
            current_node = current_node.children[index] # mise à jour du noeud courant
            self.current_position.push(current_node.move) # mise à jour de la position courante 

        leaf = current_node 

        return leaf

    '''
    Fonction expansion_backprop(mcts_nn(), leaf)

    Arguments :
        - leaf : node_nn(), noeud correspondant à la sortie de la fonction sélection(mcts_nn)

    Sorties :

    Description : 
        Cette fonction correspond à deux phases de l'algorithme MCTS : phase d'expansion/simulation et phase de rétropropagation.
        EXPANSION/SIMULATION : 
            Dans une feuille, on procède à la phase d'expansion. La position correspondant à cette feuille est évaluée par le réseau de neurones.
            On crée ensuite tous les noeuds enfants possibles (ceux correspondant à des coups légaux)
        RETROPROPAGATION :

    '''

    def expansion_backprop(self,leaf): 

        outcome = self.current_position.outcome() # issue de la partie, None si la partie n'est pas finie

        # EXPANSION

        legal_moves = self.current_position.legal_moves # génération des coups légaux

        if outcome is None: # si la partie n'est pas terminée

            dirichlet_noise, inc = dirichlet([0.03]*legal_moves.count()), 0 # bruit tiré selon une loi de dirichlet
            p,v = evaluate_position(self.model, self.current_position) # évaluation de la position à l'aide du réseau de neurones
            p = p[0]
            v = v[0,0]
            leaf.V += v # mise à jour de la valuation de la feuille

            for move in legal_moves: # création des nouveaux noeuds correspondants aux coups légaux

                if self.current_position.turn : # si c'est aux blancs de jouer
                    prob = p[self.moves_w.index(move)]

                else : # si c'est aux noirs de jouer
                    prob = p[self.moves_b.index(move)]

                prob = 0.75 * prob + 0.25 * dirichlet_noise[inc] # ajout du bruit
                inc += 1

                node_nn(move=move,parent=leaf,prob=prob) # création des noeuds enfants

            # RETROPROPAGATION

            for ancestor in leaf.iter_path_reverse(): # rétropropagation de la valuation
                ancestor.N += 1 # mise à jour du nombre de visites des ancêtres
                ancestor.V += v # mise à jour des valuations des ancêtres

        self.current_position = self.initial_position.copy() # on initialise la position courante 

        return 