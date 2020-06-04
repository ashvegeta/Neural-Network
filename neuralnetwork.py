import numpy as np

class WeightsInitialization:
    def __init__(self, L_in, L_out):
        self.L_in = L_in
        self.L_out = L_out
        self.weight=None

    def InitializeLayer(self):
        L = np.random.rand(self.L_out, self.L_in + 1)
        self.weight = L


class NeuralNet:
    def __init__(self, hidden_layers):
        self.w = []
        self.hidden_layers = hidden_layers
        self.activation = []
        self.ai = None
        self.delta = None
        self.error = None

    def Weights(self):
        for i in range(0, self.hidden_layers + 1):
            print("enter weights' details of layers", i + 1, i + 2)
            lin, lout = input("enter no of inputs and outputs:").split()
            self.w.append(WeightsInitialization(int(lin), int(lout)))

        for i in range(0, self.hidden_layers + 1):
            self.w[i].InitializeLayer()

    @staticmethod
    def SigmoidActivation(z):  # use the sigmoid activation for only classification problem
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def dSigmoid(z):  # use sigmoid derivative for classification problem only
        return z * (1 - z)

    def ForwardProp(self, X):
        self.activation = []
        self.activation.append(X)
        total_layers = self.hidden_layers + 2
        self.ai = X
        for i in range(0, total_layers - 1):
            input = self.ai
            hyp = np.dot(self.w[i].weight[:, 1:], input) + self.w[i].weight[:, 0]
            self.activation.append(hyp)
            self.ai = hyp
        return self.ai

    def BackProp(self, y, l_rate):
        self.error = y - self.ai
        for i in range(self.hidden_layers, -1, -1):
            # finding delta
            self.delta = self.error * self.activation[i + 1]

            # finding the weight gradient wrt error and updating it
            w_update = self.activation[i] * self.delta[:, np.newaxis]
            b_update = self.delta
            self.w[i].weight[:, 1:] += l_rate * w_update
            self.w[i].weight[:, 0] += l_rate * b_update

            # back-propagating error for the previous layer
            self.error = np.dot(self.w[i].weight[:, 1:].T, self.delta)

    def Train(self, X, y, iters, l_rate):
        hyp = []
        for i in range(0, len(X)):
            for j in range(0, iters):
                h = self.ForwardProp(X[i])
                self.BackProp(y[i], l_rate)
            hyp.append(h)
        return hyp


def main():
    hidden_layers = int(input("enter the no of hidden layers: "))

    nn = NeuralNet(hidden_layers)

    nn.Weights()  # random initialization of weights

    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[3, 5, 9, 12], [14, 16, 18, 20]])

    initial = []
    for i in X:
        a = nn.ForwardProp(i)
        initial.append(a)

    print("\n initial hypothesis: ", np.array(initial))

    hyp = nn.Train(X, y,  iters=600, l_rate=0.00001)

    print("\nfinal hypothesis:", np.array(hyp))
    print("\nexpected output:", y)


if __name__ == "__main__":
    main()
