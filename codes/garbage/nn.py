from keras.engine.training import Model
import json
import chess
from utils import *


def load_model():

    config_path = "/content/Reinforcement-Learning-AlphaZero/codes/model_config.json"
    weight_path = "/content/Reinforcement-Learning-AlphaZero/codes/model_weights.h5"

    with open(config_path, "rt") as f:
        model = Model.from_config(json.load(f))
        model.load_weights(weight_path)

    return model



def evaluate_position(model, position):

    input = np.array([format_input_NN(position)])
    p,v = model.predict(input)

    if not position.turn : # on prend toujours la perspective des blancs pour simplifier MCTS
        v = -v

    return p, v