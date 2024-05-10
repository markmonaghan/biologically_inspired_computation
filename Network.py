from Layer import Layer
from ActFunction import ActFunction
import re
import numpy as np

class Network:
    class InvalidInputLayer(Exception):
        def __init__(self, message):
            super().__init__(message)
    
    # Init of act.functions class
    def __init__(self):
        self.activation_functions = {"relu": ActFunction.ReLu, "sigmoid": ActFunction.Sigmoid, "sgn": ActFunction.Sgn, "tanh": ActFunction.Tanh}
    
    # Create ANN
    def create_from_str(self, layers: list[str], Yreal):
        for i, layer in enumerate(layers):
            if i == 0:
                if not re.fullmatch(r"\d+input", layer):
                    raise self.InvalidInputLayer(f"Layer \"{layer}\" doesn't match the required pattern.")
            else:
                if not re.fullmatch(r"\d+(relu|sigmoid|sgn|tanh)", layer):
                    raise self.InvalidInputLayer(f"Layer \"{layer}\" doesn't match regex")
        prev = None
        next = None
        for i, layer in enumerate(layers):
            match = re.match(r'(\d+)([a-zA-Z]+)', layer)
            if i == 0:
                self.root = Layer(n_neurons=int(match.group(1)), input_layer=True, name="root")
                prev = self.root
            else:
                next = Layer(n_neurons=int(match.group(1)), name=layer, activation_function_class=self.activation_functions[match.group(2)], prev_layer=prev)
                prev.next_layer = next
                prev = next
        self.root.get_last_layer().Yreal = Yreal
        return self
    
    def print_nn(self):
        tmp = self.root
        while (True):
            if tmp:
                print(tmp)
                tmp = tmp.next_layer
            else:
                break