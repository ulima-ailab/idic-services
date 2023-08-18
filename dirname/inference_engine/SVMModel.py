import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


class SVMmodel:
    def __init__(self):
        self.model = None

    def train_model(self, x, y, save_path=None):
        # Grid de hiperparámetros
        # ==============================================================================
        param_grid = {'C': np.logspace(-5, 7, 20)}

        # Búsqueda por validación cruzada
        # ==============================================================================
        grid = GridSearchCV(
            estimator=SVC(kernel="rbf", gamma='scale'),
            param_grid=param_grid,
            scoring='accuracy',
            n_jobs=-1,
            cv=3,
            verbose=0,
            return_train_score=True,
            refit=True
        )

        # Se asigna el resultado a _ para que no se imprima por pantalla
        _ = grid.fit(X=x, y=y)

        # Mejores hiperparámetros por validación cruzada
        # ==============================================================================
        print("----------------------------------------")
        print("Mejores hiperparámetros encontrados (cv)")
        print("----------------------------------------")
        print(grid.best_params_, ":", grid.best_score_, grid.scoring)

        self.model = grid.best_estimator_

        if save_path is not None:
            dump(self.model, save_path + ".joblib")

    def test_model(self, X, y):
        best_pred = self.model.predict(X)
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
        # Somewhere else
        self.model = load(model_path + ".joblib")
        print("Model loaded")

    def process_input(self, data):
        best_pred = self.model.predict(data)
        print(best_pred)
        return 'LP' + str(int(round(best_pred[0], ndigits=0)))
