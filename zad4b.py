import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Generowanie danych
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42,
    n_clusters_per_class=1,
)

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Normalizacja danych
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicjalizacja wag
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Warstwa ukryta
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Warstwa wyjściowa
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, X, y, output):
        # Obliczenie błędu
        y = y.reshape(-1, 1)
        error = y - output

        # Propagacja wsteczna
        d_output = error * self.sigmoid_derivative(output)
        error_hidden = d_output.dot(self.W2.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.a1)

        # Aktualizacja wag
        self.W2 += self.a1.T.dot(d_output) * self.learning_rate
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        self.W1 += X.T.dot(d_hidden) * self.learning_rate
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.loss_history = []

        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            # Obliczenie błędu
            loss = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
            self.loss_history.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return (output >= 0.5).astype(int)


# Stworzenie i trening sieci
nn = SimpleNeuralNetwork(input_size=2, hidden_size=5, output_size=1)
nn.train(X_train, y_train, learning_rate=0.1, epochs=1000)

# Ewaluacja
y_pred = nn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność na zbiorze testowym: {accuracy:.4f}")

# Wizualizacja procesu uczenia
plt.figure(figsize=(10, 6))
plt.plot(nn.loss_history)
plt.title("Wartość funkcji straty w trakcie uczenia")
plt.xlabel("Epoka")
plt.ylabel("Strata")
plt.grid(True)
plt.show()
