from sys import argv
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from Network import Network
from CostFunction import CostFunction
import numpy as np

df = pd.read_csv(argv[1])

bank = pd.read_csv("data_banknote_authentication.csv")

train_input, test_input = train_test_split(bank, test_size=0.3, random_state=1)
Yreal_train_input = train_input.iloc[:, -1].values.reshape(-1, 1)
train_input = train_input.iloc[:, :-1].values
Yreal_test_input = test_input.iloc[:, -1].values.reshape(-1, 1)
test_input = test_input.iloc[:, :-1].values

for index, row in df.iterrows():

    layers: list = ast.literal_eval(row.iloc[0].replace("-", ","))
    #Include by default a "4input" layer
    layers.insert(0, "4input")
    pso_num_particles = row.iloc[1]
    pso_num_iteration = row.iloc[2]
    omega = row.iloc[3]
    phi_p = row.iloc[4]
    phi_g = row.iloc[5]
    
    print(f"Experiment #{index}")
    print("---------------")
    print(f"Layers: {layers}")
    print(f"Num Particles PSO: {pso_num_particles}")
    print(f"Num Iterations PSO: {pso_num_iteration}")
    print(f"Omega: {omega}")
    print(f"phi_p: {phi_p}")
    print(f"phi_g: {phi_g}")
    print("")
    mse = np.empty(shape=(0,))
    seed = 100
    for repetition in range(10):
        np.random.seed(seed+repetition)
        print(f"----Experiment #{index} Repetition #{repetition}")
        
        ann = Network().create_from_str(layers=layers, Yreal=Yreal_train_input)
        ann.root.only_set_input(train_input)
        ann.root.pso(pso_num_particles, ann.root.next_layer.number_hyperparameter_ann(), pso_num_iteration, omega, phi_p, phi_g)
        ann.root.set_y_real(Yreal_test_input)
        ann.root.set_input(test_input)
        error = CostFunction.MSE.mse(ann.root.get_output_results(), Yreal_test_input)[0]
        mse = np.append(mse, error)
        print("MSE: " + str(error))
        print("%: " + str((1 - error) * 100))
        print("")
           
    print(f"Mean MSE for Experiment #{index}")
    print(np.mean(mse))
    print(f"Mean % for Experiment #{index}")
    print((1 - np.mean(mse)) * 100)
    print()
    print("##############################################")
    print("##############################################")
    print("##############################################")
    print()
    mse = np.array([])
    