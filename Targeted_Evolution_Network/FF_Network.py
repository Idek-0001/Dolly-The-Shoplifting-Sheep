import numpy as np
from copy import deepcopy
import time 
import random

class Network():
    
    def __init__ (self, layer_nums = [], fitness = 0, MUTATE_MAGNITUDE = 1.0):
        
        self.MUTATE_MAGNITUDE = MUTATE_MAGNITUDE

        self.layers = len(layer_nums)
        self.hidden_layers = self.layers - 2

        self.fitness = fitness
        self.outputs = [0.0 for o in range(layer_nums[self.layers-1])]

        self.biases = ([[random.uniform(-1.0,1.0) for node in range(layer_nums[layer + 1])] for layer in range(self.hidden_layers + 1)])
        self.bias_fitnesses = ([[0.0 for node in range(layer_nums[layer + 1])] for layer in range(self.hidden_layers + 1)])
        self.weights = ([[[random.uniform(-1.0,1.0) for nextNode in range(layer_nums[layer + 1])] for node in range(layer_nums[layer])] for layer in range(self.hidden_layers + 1)])
        self.weight_fitnesses = ([[[0.0 for nextNode in range(layer_nums[layer + 1])] for node in range(layer_nums[layer])] for layer in range(self.hidden_layers + 1)])

    def sigmoid(self, value):
        return 1.0/ (1.0 + np.exp(-value))
    
    def get_layer_outputs(self, inputs, layer):
        return self.sigmoid(np.dot(inputs, self.weights[layer]) + self.biases[layer])

    def propogate(self, inputs = [], current_layer = 0):
        if current_layer == self.hidden_layers:
            self.outputs = self.get_layer_outputs(inputs, current_layer) 
        else:
            self.propogate(self.get_layer_outputs(inputs, current_layer) , current_layer= current_layer + 1)
    
    def compute_weight_impacts(self, value, weights):
        total_sum = sum(weights)
        impact_list = np.multiply(weights, value/total_sum)
        return impact_list

    def assign_impacts(self, values, current_layer):
        if (current_layer < 0): return          #   Base Case

        bias_impacts = np.multiply(self.biases[current_layer], map(sum(),values)/sum(self.biases[current_layer]))

        for node, bias_impact in enumerate(bias_impacts):
            self.weight_fitnesses = [x + y for x,y in zip(self.weight_fitnesses[current_layer][node], self.compute_impacts(bias_impact, self.weights[current_layer][node]))]
        
        self.bias_fitnesses[current_layer] = [x + y for x,y in zip(self.bias_fitnesses, bias_impacts)]

        return np.dot(bias_impacts, self.weights[current_layer])

    def mutate(self, error_value, weight_fitness_threshold = 0, bias_fitness_threshold = 0, current_layer = 0):
        
        if (current_layer == self.hidden_layers + 1): return            #   Base Case

        error_value = self.assign_impacts(error_value, current_layer) #Updates weight and bias fitnesses

        #Mutate biases
        biases_to_mutate = deepcopy(self.bias_fitnesses[current_layer])
        biases_to_mutate[current_layer][self.bias_fitnesses[current_layer] > bias_fitness_threshold] = 0
        bias_mutate_amounts = np.random.uniform(low= -1.0 * self.MUTATE_MAGNITUDE, high= 1.0 * self.MUTATE_MAGNITUDE, size= (len(biases_to_mutate),))
        bias_mutations = np.multiply(bias_mutate_amounts, biases_to_mutate)
        self.biases[current_layer] = [x + y for x,y in zip(self.biases[current_layer], bias_mutations)]

        #Mutate weights
        for node in range(len(self.weights[current_layer])):
            weights_to_mutate = deepcopy(self.weight_fitnesses[current_layer][node])
            weights_to_mutate [weights_to_mutate > weight_fitness_threshold] = 0
            weight_mutate_amounts = np.random.uniform(low= -1.0 * self.MUTATE_MAGNITUDE, high= 1.0 * self.MUTATE_MAGNITUDE, size= (len(weights_to_mutate),))
            weight_mutations = np.multiply(weight_mutate_amounts, weights_to_mutate)
            self.weights[current_layer][node] = [x + y for x,y in zip(self.weights[current_layer][node], weight_mutations)]
        
        self.mutate(error_value, weight_fitness_threshold= weight_fitness_threshold, bias_fitness_threshold= bias_fitness_threshold, current_layer = current_layer + 1)

def threshold_curve(value):
    return value

def determine_bias_threshold(network):
    #network_accuracy = network.fitness / max_network_fitness

    bias_max = np.mean(max(network.bias_fitnesses))
    bias_min = np.mean(min(network.bias_fitnesses))
    bias_average = np.mean(network.bias_fitnesses)

    bias_threshold = bias_average * threshold_curve(bias_max - bias_min)
    return bias_threshold

def determine_weight_threshold(network):
    #network_accuracy = network.fitness / max_network_fitness

    weight_max = np.mean(max(network.weight_fitnesses))
    weight_min = np.mean(min(network.weight_fitnesses))
    weight_average = np.mean(network.weight_fitnesses)
    weight_threshold = weight_average * threshold_curve(weight_max - weight_min)

    return weight_threshold




n = Network([2,2,2])

n.mutate([1,0], weight_fitness_threshold= determine_weight_threshold(n), bias_fitness_threshold= determine_bias_threshold(n))