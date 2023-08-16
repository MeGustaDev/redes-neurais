import numpy as np

class Perceptron(object):

    def __init__(self, learning_rate=0.01, epochs=50):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def _sum(self, input):
        response = np.dot(input, self._weights) + self.bias

    def activation_function(self, value):
        #função degrau
        # se o valor for maior ou igual a zero retorna 1
        # senão retorna -1
        return np.where(value >= 0.0, 1, -1)
    
    def fit(self, X, y):
        """
        Vamos começar com os pesos como um array
        numpy aleatório, já que não temos pistas sobre
        quais são os pesos corretos.
        """
        self._bias = np.random.uniform(-1, 1)
        self._weights = np.random.uniform(-1,1, (X.shape[1]))

        for epoch_number in range(self.epochs):
            for xi, target in zip(X, y):
                output = self.predict(xi)
                update = self.learning_rate * (target - output)
                self._bias += update
                self._weights += update * xi
        return self
    

    def predict(self, X):
        result = self.activation_function(self._sum(X))

        return result
