import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Wczytaj dane Iris (długości i szerokości działek oraz płatków)
X = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2],
    [5.4, 3.9, 1.7, 0.4],
    [4.6, 3.4, 1.4, 0.3],
    [5.0, 3.4, 1.5, 0.2],
    [4.4, 2.9, 1.4, 0.2],
    [4.9, 3.1, 1.5, 0.1]
])

y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) # Odpowiednie etykiety dla gatunków

# Podziel dane na zbiory treningowe i testowe
train_size = int(0.8 * len(X))
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# Funkcja aktywacji (skok Heaviside'a)
def step_function(x):
    return 1 if x >= 0 else 0

# Inicjalizacja wag i progu
weights = np.zeros(X.shape[1])
threshold = 0
learning_rate = 0.01

# Trenuj perceptron
# ZAD1 dokladnosc wzrasta
for epoch in range(100):
    for i in range(len(X_train)):
        # Oblicz sumę ważoną
        weighted_sum = np.dot(weights, X_train[i]) - threshold
        # Oblicz błąd
        error = y_train[i] - step_function(weighted_sum)
        # Aktualizuj wagi i próg
        weights += learning_rate * error * X_train[i]
        threshold -= learning_rate * error

# Dokonaj przewidywań dla danych testowych
y_pred = []
for x in X_test:
    weighted_sum = np.dot(weights, x) - threshold
    y_pred.append(step_function(weighted_sum))

y_pred = np.array(y_pred)

# Oblicz dokładność klasyfikacji
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f'Dokładność modelu perceptronu: {accuracy * 100:.2f}%')

# ZAD3
plt.plot(y_pred, label='Przewidywane')
plt.plot(y_test, label='Rzeczywiste', linestyle='dashed')
plt.xlabel('Próba')
plt.ylabel('Wynik')
plt.legend()
plt.show()