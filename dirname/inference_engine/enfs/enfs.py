import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from .diffevo import differential_evolution
from .anfis import ANFIS
from .fobj import *


class ENFS:
    # Evaluates the objective function
    def __init__(self):
        self.fis = None
        self.lbls = None
        self.data = None

    def __eval_objective(self, params):
        # From the parameter vector (genome) gets each set of parameters (means, standard deviations and sequent singletons)
        mus = params[0:self.fis.m * self.fis.n]
        sigmas = params[self.fis.m * self.fis.n:2 * self.fis.m * self.fis.n]
        y = params[2 * self.fis.m * self.fis.n:]
        # Sets the FIS parameters to the ones on the genome
        self.fis.setmfs(mus, sigmas, y)
        pred = self.fis.infer(self.data)
        loss = 1 - nse(pred, self.lbls)
        return loss

    def train_model(self, x, y, dim_x, n_rules, save_path=None):
        self.data = x
        self.lbls = y

        # Creates the inference system
        self.fis = ANFIS(dim_x, n_rules)
        n_params = 2 * (n_rules * dim_x) + n_rules  # Total number of parameters (genome size)

        # Runs the evolution cycle
        start_time = time.time()
        result = list(differential_evolution(self.__eval_objective, bounds=[(-2, 2)] * n_params, gens=10))
        end_time = time.time()
        print('Evolution time: %f' % (end_time - start_time))
        # Gets the last genome
        best_params = result[-1][0]
        best_mus = best_params[0:self.fis.m * self.fis.n]
        best_sigmas = best_params[self.fis.m * self.fis.n:2 * self.fis.m * self.fis.n]
        best_y = best_params[2 * self.fis.m * self.fis.n:2 * self.fis.m * self.fis.n + self.fis.m]

        if save_path is not None:
            np.savetxt(save_path + "_1.txt", best_mus, delimiter=';')
            np.savetxt(save_path + "_2.txt", best_sigmas, delimiter=';')
            np.savetxt(save_path + "_3.txt", best_y, delimiter=';')

            file = open(save_path + ".txt", "w")
            file.write(str(dim_x) + "\n")
            file.write(str(n_rules) + "\n")
            file.close()
        """
        # Sets the FIS parameters to the ones of the last genome
        self.fis.setmfs(best_mus, best_sigmas, best_y)
        # Predicts output for the training set
        best_pred = self.fis.infer(self.data)
        # Plots the real and predicted one series
        plt.plot(y)
        plt.plot(best_pred)
        plt.legend(['Real', 'Predicted'])
        plt.show()
        """
        print('Best fitness: %f' % result[-1][1])

    def test_model(self, X, y):
        best_pred = self.fis.infer(X)
        accuracy = accuracy_score(
            y_true=y,
            y_pred=best_pred,
            normalize=True
        )
        print("")
        print(f"El accuracy de test es: {100 * accuracy}%")

        # Plots the real and predicted one series
        plt.plot(y)
        plt.plot(best_pred)
        plt.legend(['Real', 'Predicted'])
        plt.show()

    def load_model(self, model_path):
        with open(model_path + ".txt") as f:
            dim_x, n_rules = f.readlines()
        best_mus = np.loadtxt(model_path + "_1.txt")
        best_sigmas = np.loadtxt(model_path + "_2.txt")
        best_y = np.loadtxt(model_path + "_3.txt")

        self.fis = ANFIS(int(dim_x), int(n_rules))
        self.fis.setmfs(best_mus, best_sigmas, best_y)
        print("Model loaded")

    def process_input(self, data):
        best_pred = self.fis.infer(data)
        return 'LP' + str(int(round(best_pred[0], ndigits=0)))
