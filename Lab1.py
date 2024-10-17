# Implementacja perceptronu uczącego się klasyfikacji AND w Pythonie
import numpy as np
import matplotlib.pyplot as plt
errors = []
# Dane treningowe
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # Wejścia (dane treningowe)
X = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]])
#y = np.array([0, 0, 1, 1])
# y = np.array([0, 0, 0, 1]) # Oczekiwane wyjścia (AND)
# ZAD1
y = np.array([0, 0, 1, 1])
# Parametry perceptronu
w = np.random.rand(3) # Inicjalizacja wag (losowe wartości początkowe)
b = np.random.rand(1) # Inicjalizacja biasu (losowa wartość początkowa)
# eta = 0.1 # Współczynnik uczenia (determinuje, jak duże są kroki aktualizacji wag)
# ZAD5
eta = 0.01  # Mniejsza wartość
# epochs = 10 # Liczba epok (ile razy przechodzimy przez dane)
# ZAD2
epochs = 100
# Funkcja aktywacji Funkcja aktywacji odpowiada za przekształcenie sumy ważonej na wynik, który neuron może zwrócić.
# W przypadku perceptronu jest to najczęściej funkcja skoku jednostkowego, która zwraca 1, jeśli wartość wejściowa
# jest większa od 0, a 0 w przeciwnym razie.
def activation_function(z):
    return 1 if z > 0 else 0
# ZAD3
def relu(z):
    return max(0, z)
# Algorytm uczenia perceptronu
for epoch in range(epochs):
    total_error = 0
    for i in range(len(X)):
        # Obliczenie sumy ważonej (z)
        z = np.dot(X[i], w) + b # Zastosowanie iloczynu skalarnego wejść i wag oraz dodanie biasu

        # Zastosowanie funkcji aktywacji
        # Funkcja aktywacji decyduje, czy neuron "odpali" (zwróci 1), czy nie (zwróci 0).
        # Dzięki temu perceptron może klasyfikować dane na podstawie obliczonej sumy ważonej.
        y_pred = relu(z) # ZAD3 # activation_function(z)

        # Obliczenie błędu (różnica między oczekiwanym a przewidywanym wyjściem)
        error = y[i] - y_pred
        total_error += abs(error)

        # Aktualizacja wag i biasu w celu zminimalizowania błędu
        w += eta * error * X[i] # Wagi są aktualizowane proporcjonalnie do błędu, współczynnika uczenia i wejść
        b += eta * error # Bias jest aktualizowany proporcjonalnie do błędu i współczynnika uczenia
    # ZAD4
    errors.append(total_error)
# Wyświetlenie wyuczonych wag i biasu
print("Wyuczone wagi:", w)
print("Wyuczony bias:", b)
plt.plot(errors)
plt.xlabel('Epoch')
plt.ylabel('Total Error')
plt.show()