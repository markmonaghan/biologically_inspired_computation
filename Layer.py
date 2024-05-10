from __future__ import annotations
from scipy import stats
import numpy as np
from CostFunction import CostFunction
from uuid import uuid4

# Layer class
# Is going to store all data form the layer
class Layer:
    def __init__(self,n_neurons, name, bias = None, activation_function_class = None, weights = None, prev_layer = None, next_layer = None, input_layer = False, Yreal = None):
        self.n_neurons = n_neurons
        self.input_layer = input_layer
        self.name = name
        self.uuid = uuid4()
        self.activation_function_class = activation_function_class
        self.perceptrons: list
        self.prev_layer: Layer = prev_layer
        self.next_layer: Layer = next_layer 
        self.bias = bias
        self.weights = weights
        self.weights_temp = None
        self.output = None
        self.Yreal = Yreal
        self.delta = None
        
        if self.bias is None and not self.input_layer:
            self.bias = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size=self.n_neurons).reshape(1,self.n_neurons),3)
        if self.weights is None and not self.input_layer:
            self.weights = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size=self.n_neurons*self.prev_layer.n_neurons).reshape(self.prev_layer.n_neurons,self.n_neurons), 3)
        if self.activation_function_class:
            self.activation_function = getattr(self.activation_function_class, self.activation_function_class.__name__.lower())
            self.activation_function_derivative = getattr(self.activation_function_class, f"{self.activation_function_class.__name__.lower()}_derivative")

    def set_prev_layer(self, prev_layer):
        self.prev_layer = prev_layer

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    def set_input(self, input: list[list]):
        self.output = input
        self.next_layer.forward_layer(input)
        
    def only_set_input(self, input: list[list]):
        self.input = input
        
    def forward_from_root(self):
        self.next_layer.forward_layer(self.input)
    
    def forward_layer(self, input: list[list]):
        
        z = input @ self.weights
        z = self.activation_function(z)
        self.output = z
        if self.next_layer:
            self.next_layer.forward_layer(z)
            
        # Code to use backpropagation instead of PSO
        #else:
            #print(f"Last Layer: {self.name}")
            #print("-----BACKPROPAGATION STARTING NOW-----")
            #print(f"MSE: {CostFunction.MSE.mse(z, self.Yreal)[0]}")
            #self.backpropagtion()
            #print(self.pso(100, self.get_root_layer().next_layer.number_hyperparameter_ann(), 100))
            #print(f"output: {z}")
        #print(f"end forward: {self.name}")
    
    #We need to call this function from root
    def assign_new_values(self, new_values: list):
        num_filas = len(self.weights)
        num_columnas = len(self.weights[0])

        for col in range(num_columnas):
            for row in range(num_filas):
                self.weights[row][col] = new_values.pop(0)
                
        for b in range(len(self.bias[0])):
            self.bias[0][b] = new_values.pop(0)
        
        # Code to optimize the Act. Function for each layer, sometimes it's not working properly
        af = new_values.pop(0)
        #print(af)
        #if (af <= 1):
        #    self.activation_function = ActFunction.Sigmoid.sigmoid
        #elif (af > 1 and af <= 2):
        #    self.activation_function = ActFunction.ReLu.relu
        #elif (af > 2 and af <= 3):
        #    self.activation_function = ActFunction.Sgn.sgn
        #elif (af > 3):
        #    self.activation_function = ActFunction.Tanh.tanh
        #else:
        #    print(f"[ERROR] ACTFUNCT {af}")
        #    exit(1)
                
        if self.next_layer:
            self.next_layer.assign_new_values(new_values=new_values)
    
    #We need to call this function from root too
    def pso(self, num_particles, num_params, num_iterations, omega, phi_p, phi_g):
        particle_position = np.random.rand(num_particles, num_params)
        particle_velocity = np.random.rand(num_particles, num_params)

        best_particle_pos = particle_position.copy()
        best_particle_cost = np.array([np.inf] * num_particles)
        best_global_position = particle_position[0].copy()
        best_global_cost = np.inf
        actual_cost = 0
        
        for _ in range(num_iterations):
            for i in range(num_particles):
                self.next_layer.assign_new_values(particle_position[i].tolist())
                self.forward_from_root()

                actual_cost = CostFunction.MSE.mse(self.get_output_results(), self.get_last_layer().Yreal)[0]

                # Update best particle position
                if actual_cost < best_particle_cost[i]:
                    best_particle_cost[i] = actual_cost
                    best_particle_pos[i] = particle_position[i].copy()

                # Update best global position
                if actual_cost < best_global_cost:
                    best_global_cost = actual_cost
                    best_global_position = particle_position[i].copy()

            ################################
            ################################
            ## Omega = Inertia Factor     ##
            ## phi_p = Cognitive Factor   ##
            ## phi_g = Social Factor      ##
            ################################
            ################################

            # Loop to update velocity and positions
            for i in range(num_particles):
                # Random values for personal and global coef.
                r_p, r_g = np.random.rand(), np.random.rand()
                actual_velocity = (omega * particle_velocity[i] +
                                   phi_p * r_p * (best_particle_pos[i] - particle_position[i]) +
                                   phi_g * r_g * (best_global_position - particle_position[i]))
                particle_velocity[i] = actual_velocity
                particle_position[i] = particle_position[i] + actual_velocity

        return best_global_position

    def number_hyperparameter_ann(self):
        num = self.weights.shape[0] * self.weights.shape[1] + self.n_neurons + 1
        
        if self.next_layer:
            num += self.next_layer.number_hyperparameter_ann()
        
        return num
    
    #Not used in PSO algorithm, (coded before know the subject)
    def backpropagtion(self):
        a = self.output
        if not self.next_layer:
            self.delta = CostFunction.MSE.mse(a, self.Yreal)[1] * self.activation_function_derivative(a)
        else:
            self.delta = self.next_layer.delta @ self.next_layer.weights_temp * self.activation_function_derivative(a)
        self.weights_temp = self.weights.transpose()
        self.bias = self.bias - self.delta.mean() * 0.1
        beta = (self.prev_layer.output.T @ self.delta) * 0.1
        self.weights = self.weights - beta

        if self.prev_layer.prev_layer:
            self.prev_layer.backpropagtion()
        
    def get_output_results(self):
        if not self.next_layer:
            return self.output
        else:
            return self.next_layer.get_output_results()
        
    def get_last_layer(self):
        if not self.next_layer:
            return self
        else:
            return self.next_layer.get_last_layer()
        
    def get_root_layer(self):
        if not self.prev_layer:
            return self
        else:
            return self.prev_layer.get_root_layer()
        
    def set_y_real(self, Yreal):
        self.get_last_layer().Yreal = Yreal
            
    def __repr__ (self):
        if not self.input_layer:
            return f"-------------------\nLayer {self.uuid}:\nn_neurons={self.n_neurons}\nact_function={self.activation_function.__name__}\n"
        return f"-------------------\nInput Layer {self.uuid}:\nn_neurons={self.n_neurons}\n"