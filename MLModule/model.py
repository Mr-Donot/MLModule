from random import random
import math
from typing import Union
import pickle, os


class NeuralNetwork():

    def __init__(self, nb_input: int, nb_output: int, hidden_layers: list[int]=[]):
        self.nb_input = nb_input
        self.nb_output = nb_output
        self.hidden_layers = hidden_layers
        self.nb_layers = 2 + len(hidden_layers)
        self.init_network()

    def init_network(self):
        network = []
        layers_values = []
        self.layers_length = []
        self.layers_length.append(self.nb_input)
        for hidden in self.hidden_layers:
            self.layers_length.append(hidden)
        self.layers_length.append(self.nb_output)

        #we don't create anything for the input layer, so len() -1 and [i+1]
        for i in range(len(self.layers_length)-1):
            layer = []
            layer_values = []
            for output in range(self.layers_length[i+1]):
                layer_values.append(0)
                bias = random()
                w_output = []
                for input in range(self.layers_length[i]):
                    w_output.append(random()*2 - 1) #value between [-1, 1]
                layer.append({"weights":w_output, "bias":bias, "cost_grad_w" : [0]*len(w_output), "cost_grad_b" : 0})
            network.append(layer)                
            layers_values.append(layer_values)

        self.network = network
        self.layers_values = layers_values


    def print_network(self):

        idx_layer = 0

        for w_layer in (self.network):
            idx_layer += 1
            print(f"Layer {idx_layer} : ")
            idx_input = 0
            for w_output  in (w_layer):
                print(f"output {idx_input} : {w_output}")
                idx_input += 1



    def predict(self, input : list[float]) -> Union[None, list[float]]:
        
        if len(input) != self.nb_input:
            print(f"Error, not the good input length ({len(input)}), should be {self.nb_input}")
            return None
        
        idx_layer = 0

        while idx_layer != self.nb_layers-1 :
            next_layer = self.calculate_layer(idx_layer, input)
            input = next_layer
            self.layers_values[idx_layer] = next_layer
            idx_layer += 1



        return next_layer


    def calculate_layer(self, idx_layer: int, input : list[float]) -> list[float]:
        nb_values_in = len(input)
        nb_values_out = self.layers_length[idx_layer+1]
        new_layer = []

        for out in range(nb_values_out):
            value = 0
            for inp in range(nb_values_in):
                value += self.network[idx_layer][out]["weights"][inp] * input[inp]
            value += self.network[idx_layer][out]["bias"]
            new_layer.append(self.activation_function(value))

        return new_layer
    
    def activation_function(self, value: float) -> float:
        return 1 / (1 + math.exp(- value))
    
    def activation_function_deriv(self, value: float) -> float:
        activated_value = self.activation_function(value)
        return activated_value * (1 - activated_value)

    def node_cost(self, output: float, expected_output: float) -> float:
        return (output - expected_output) ** 2
    
    def node_cost_deriv(self, output: float, expected_output: float) -> float:
        return 2 * (output - expected_output)
    
    def datapoint_cost(self, X: list[float], y: list[float])-> float:
        output = self.predict(X)
        cost: float = 0.0
        for i in range(len(output)):
            cost += self.node_cost(output[i], y[i])
        return cost
    
    def cost(self, Xs: list[list[float]], ys: list[list[float]])-> float:
        total_cost: float = 0.0

        for x, y in zip(Xs, ys):
            total_cost += self.datapoint_cost(x, y)

        return total_cost

    def apply_gradient_descent(self, learning_rate: float):
        for layer in self.network:
            for output in range(len(layer)):
                
                layer[output]["bias"] -= layer[output]["cost_grad_b"] * learning_rate 
                for input in range(len(layer[output]["weights"])):
                    layer[output]["weights"][input] -= layer[output]["cost_grad_w"][input] * learning_rate

    def update_gradients(self, idx_layer_out: int, node_values : list[float]):

        for n_out in range(len(self.network[idx_layer_out])):

            for n_in in range(len(self.network[idx_layer_out-1])):
                derivative_cost_weight = self.layers_values[idx_layer_out-1][n_in] * node_values[n_out]
                self.network[idx_layer_out][n_out]["cost_grad_w"][n_in] += derivative_cost_weight

            derivative_cost_bias = 1 * node_values[n_out]
            self.network[idx_layer_out][n_out]["cost_grad_b"] += derivative_cost_bias


    def calculate_output_node_values(self, output: list[float], expected_output: list[float]) -> list[float]:
        node_values = []
        for i in range(len(expected_output)):
            cost_derivative = self.node_cost_deriv(output[i], expected_output[i])
            weighted_input = self.layers_values[-2][i]
            activation_derivative = self.activation_function_deriv(weighted_input)
            node_values.append(activation_derivative * cost_derivative)
        
        return node_values

    def calculate_hidden_node_values(self, idx_layer:int, old_node_values: list[float]) -> list[float]:

        #change old layer to simply found it in self.network[idx_layer+1]
        new_node_values = []

        for new_idx in range(len(self.network[idx_layer])):
            new_node_value = 0
            for old_idx in range(len(old_node_values)):
                weighted_input_deriv = self.network[idx_layer+1][old_idx]["weights"][new_idx]
                new_node_value += weighted_input_deriv * old_node_values[old_idx]
            
            new_node_value *= self.activation_function_deriv(self.layers_values[idx_layer][new_idx])
            new_node_values.append(new_node_value)

        return new_node_values


    def update_all_gradients(self, datapoint: list[float], expected_output: list[float]):

        output = self.predict(datapoint)

        node_values = self.calculate_output_node_values(output, expected_output)

        self.update_gradients(-1, node_values)

        for hidden_idx in range(len(self.network)-1, -1):
            node_values = self.calculate_hidden_node_values(hidden_idx, node_values)
            self.update_gradients(hidden_idx, node_values)

    def clean_all_gradients(self):
        for layer in range(len(self.network)):
            for output in range(len(self.network[layer])):
                self.network[layer][output]["cost_grad_b"] = 0
                self.network[layer][output]["cost_grad_w"] = [0]*len(self.network[layer][output]["cost_grad_w"])

    def learn2(self, Xs: list[list[float]], ys : list[list[float]], learning_rate: float = 0.001):

        for x, y in zip(Xs, ys):
            self.update_all_gradients(x, y)

        self.apply_gradient_descent(learning_rate)  # divide lr by len(Xs ???)

        self.clean_all_gradients()

    def learn(self, Xs: list[list[float]], ys : list[list[float]], learning_rate: float = 0.001):
        h = 0.00001
        original_cost = self.cost(Xs, ys)

        for layer in self.network:
            for output in range(len(layer)):
                layer[output]["bias"] += h
                delta_cost = self.cost(Xs, ys) - original_cost
                layer[output]["bias"] -= h
                layer[output]["cost_grad_b"] = delta_cost / h
                
                for input in range(len(layer[output]["weights"])):
                    layer[output]["weights"][input] += h
                    delta_cost = self.cost(Xs, ys) - original_cost
                    layer[output]["weights"][input] -= h
                    layer[output]["cost_grad_w"][input] = delta_cost / h

        self.apply_gradient_descent(learning_rate)
        

    def save_weights(self, path="model/points_2_dim.pkl"):
        directory = os.path.dirname(path)

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, 'wb') as file:
            pickle.dump(self.network, file)

    def load_weights(self, path="model/points_2_dim.pkl"):
        try:
            with open(path, 'rb') as file:
                self.network = pickle.load(file)
                print("Weights loaded successfully.")
        except FileNotFoundError:
            print(f"File {path} not found. Initializing a new network.")
            self.init_network()