# Implementacja perceptronu uczącego się klasyfikacji AND w Pythonie
import numpy as np
# Dane treningowe
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # Wejścia (dane treningowe)
y = np.array([0, 0, 0, 1]) # Oczekiwane wyjścia (AND)
# Parametry perceptronu
w = np.random.rand(2) # Inicjalizacja wag (losowe wartości początkowe)
b = np.random.rand(1) # Inicjalizacja biasu (losowa wartość początkowa)
eta = 0.1 # Współczynnik uczenia (determinuje, jak duże są kroki aktualizacji wag)
epochs = 10 # Liczba epok (ile razy przechodzimy przez dane)
# Funkcja aktywacji Funkcja aktywacji odpowiada za przekształcenie sumy ważonej na wynik, który neuron może zwrócić.
# W przypadku perceptronu jest to najczęściej funkcja skoku jednostkowego, która zwraca 1, jeśli wartość wejściowa
# jest większa od 0, a 0 w przeciwnym razie.
def activation_function(z):
    return 1 if z > 0 else 0
# Algorytm uczenia perceptronu
for epoch in range(epochs):
    for i in range(len(X)):
        # Obliczenie sumy ważonej (z)
        z = np.dot(X[i], w) + b # Zastosowanie iloczynu skalarnego wejść i wag oraz dodanie biasu

        # Zastosowanie funkcji aktywacji
        # Funkcja aktywacji decyduje, czy neuron "odpali" (zwróci 1), czy nie (zwróci 0).
        # Dzięki temu perceptron może klasyfikować dane na podstawie obliczonej sumy ważonej.
        y_pred = activation_function(z)

        # Obliczenie błędu (różnica między oczekiwanym a przewidywanym wyjściem)
        error = y[i] - y_pred

        # Aktualizacja wag i biasu w celu zminimalizowania błędu
        w += eta * error * X[i] # Wagi są aktualizowane proporcjonalnie do błędu, współczynnika uczenia i wejść
        b += eta * error # Bias jest aktualizowany proporcjonalnie do błędu i współczynnika uczenia
# Wyświetlenie wyuczonych wag i biasu
print("Wyuczone wagi:", w)
print("Wyuczony bias:", b)