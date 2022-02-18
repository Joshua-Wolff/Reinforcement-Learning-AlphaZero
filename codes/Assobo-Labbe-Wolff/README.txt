
README.txt


PROJET REINFORCEMENT LEARNING

Auteurs : Kévin Assobo - Robin Labbé - Joshua Wolff


LES FICHIERS :

	- mcts.ipynb : notebook qui implémente la méthode mcts pour les échecs
	- model_weights.h5 : poids du réseau de neurones pré-entraîné
	- model_config.json : fichier de configuration du réseau de neurones pré-entraîné


DÉMARCHE POUR TESTER LE CODE :

	- Ouvrir mcts.ipynb sur google colab

	- WARNING : S'assurer qu'une instance GPU est disponible (modifier -> paramètres du notebook -> Accélérateur matériel = GPU). En effet le réseau de neurones ne peut pas être évalué sur CPU pour des raisons de formatage des couches convolutionnelles.

	- Ajouter les deux fichiers model_weights.h5 et model_config.json dans l'espace de stockage de la session (symbole dossier dans la barre latérale gauche -> Importer dans l'espace de stockage de la session)

	- Il ne reste plus qu'à lancer les cellules du notebook ! 
	

