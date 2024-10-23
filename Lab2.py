import numpy as np
# ZAD2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

# ZAD2 Min-Max Normalizacja skaluje wartości tak, że wszystkie mieszczą się w przedziale od 0 do 1, co pozwala
# modelowi na bardziej efektywne trenowanie. Standaryzacja przesuwa dane, aby ich średnia wynosiła 0, a odchylenie
# standardowe było równe 1. Działa to dobrze w przypadkach, gdy dane mają rozbieżne skale. scaler = MinMaxScaler()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Funkcja aktywacji (skok Heaviside'a)
def step_function(x):
    return 1 if x >= 0 else 0

# Inicjalizacja wag i progu
weights = np.zeros(X.shape[1])
threshold = 0
learning_rate = 0.01

# ZAD4
errors = []

# Trenuj perceptron
# ZAD1 dokladnosc wzrasta
for epoch in range(100):
    total_error = 0
    for i in range(len(X_train)):
        # Oblicz sumę ważoną
        weighted_sum = np.dot(weights, X_train[i]) - threshold
        # Oblicz błąd
        error = y_train[i] - step_function(weighted_sum)
        total_error += abs(error)
        # Aktualizuj wagi i próg
        weights += learning_rate * error * X_train[i]
        threshold -= learning_rate * error
    errors.append(total_error)
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
# plt.plot(y_pred, label='Przewidywane')
# plt.plot(y_test, label='Rzeczywiste', linestyle='dashed')
# plt.xlabel('Próba')
# plt.ylabel('Wynik')
# plt.legend()
# plt.show()

# ZAD4
plt.plot(errors)
plt.xlabel('Epoka')
plt.ylabel('Suma błędów')
plt.title('Suma błędów w każdej epoce treningowej')
plt.show()