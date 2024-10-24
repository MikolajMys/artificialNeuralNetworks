import numpy as np
import matplotlib.pyplot as plt  # Import do rysowania wykresu

# Definiujemy funkcję aktywacji sigmoid oraz jej pochodną
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Pochodna sigmoidu w zależności od wyjścia funkcji sigmoid
    return x * (1 - x)

# Dane wejściowe (wektory wejściowe)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Oczekiwane wyjścia (etykiety)
y = np.array([[0],
              [1],
              [1],
              [0]])

# Ustawiamy ziarno losowe dla powtarzalności wyników
np.random.seed(1)

# Inicjalizacja wag z losowymi wartościami
weights_input_hidden = 2 * np.random.random((2, 2)) - 1  # Wagi między warstwą wejściową a ukrytą
weights_hidden_output = 2 * np.random.random((2, 1)) - 1  # Wagi między warstwą ukrytą a wyjściową

# Współczynnik uczenia
learning_rate = 0.1

# Liczba iteracji treningowych
epochs = 10000

# Lista do zapisywania średnich błędów
errors = []

# Główna pętla treningowa
for epoch in range(epochs):
    # **Forward propagation**
    # Obliczamy wejście i wyjście warstwy ukrytej
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    # Obliczamy wejście i wyjście warstwy wyjściowej
    final_input = np.dot(hidden_layer_output, weights_hidden_output)
    final_output = sigmoid(final_input)

    # **Obliczanie błędu**
    error = y - final_output

    # Zapisz średni błąd dla każdej epoki
    mean_error = np.mean(np.abs(error))
    errors.append(mean_error)

    # **Wsteczna propagacja błędu**
    # Obliczamy gradient dla warstwy wyjściowej
    d_final_output = error * sigmoid_derivative(final_output)

    # Obliczamy błąd dla warstwy ukrytej
    error_hidden_layer = d_final_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # **Aktualizacja wag**
    weights_hidden_output += hidden_layer_output.T.dot(d_final_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

    # Opcjonalnie, co 1000 epok, wyświetlamy bieżącą wartość błędu
    if (epoch % 1000) == 0:
        print(f'Epoka {epoch}, strata: {mean_error}')

# **Testowanie sieci po treningu**
print('\nWyniki po treningu:')
hidden_layer_input = np.dot(X, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)
final_input = np.dot(hidden_layer_output, weights_hidden_output)
final_output = sigmoid(final_input)
print(final_output)

# **Rysowanie wykresu błędu w zależności od liczby epok**
# ZAD3
plt.plot(errors)
plt.xlabel('Epoka')
plt.ylabel('Średnia wartość błędu')
plt.title('Średnia wartość błędu w zależności od liczby epok')
plt.show()
