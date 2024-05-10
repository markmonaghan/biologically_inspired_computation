import numpy as np

class CostFunction:

    # Mean Squared Error
    class MSE:
        def mse(Ypredich, Yreal):
            # Calc error
            x = (np.array(Ypredich) - np.array(Yreal)) ** 2
            x = np.mean(x)
            # Calc deriv funct
            y = np.array(Ypredich) - np.array(Yreal)
            return (x,y)