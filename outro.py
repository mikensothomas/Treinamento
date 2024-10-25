# Importando bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Dados da tabela (Proteína, Açúcar, Gordura, Fibra)
X = np.array([
    [25, 2, 5, 0],   # Salmão grelhado
    [30, 1, 3, 0],   # Peito de frango
    [12, 4, 2, 1],   # Iogurte natural
    [8, 15, 6, 0],   # Queijo cottage
    [6, 50, 20, 0],  # Donut
    [4, 45, 18, 0],  # Bolo de chocolate
    [3, 60, 0, 0],   # Refrigerante diet
    [5, 38, 10, 1],  # Barrinha de cereal
    [10, 10, 1, 8],  # Lentilhas cozidas
    [7, 25, 15, 0]   # Batata frita
])

# Saídas esperadas: 1 = Saudável, 0 = Não saudável
y = np.array([
    [1],  # Salmão grelhado
    [1],  # Peito de frango
    [1],  # Iogurte natural
    [1],  # Queijo cottage
    [0],  # Donut
    [0],  # Bolo de chocolate
    [0],  # Refrigerante diet
    [0],  # Barrinha de cereal
    [1],  # Lentilhas cozidas
    [0]   # Batata frita
])

# Funções de ativação e suas derivadas
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Função para calcular o erro quadrático médio (MSE)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gráfico do Erro Quadrático Médio (MSE) durante o treinamento
def plot_training_curve(epochs, mse_values):
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Erro quadrático médio ao longo das épocas
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), mse_values, label='MSE')
    plt.title('Erro Quadrático Médio (MSE) durante o Treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.grid(True)

# Gerar gráfico de separação de classes
def plot_decision_boundary(X, y, mlp, resolution=0.01):
    # Criando um grid de valores para fazer previsões
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    Z = mlp.feedforward(np.c_[xx.ravel(), yy.ravel(), np.zeros((xx.ravel().shape[0], 2))])
    Z = Z.reshape(xx.shape)

    # Plot da separação
    plt.subplot(1, 2, 2)
    plt.contourf(xx, yy, Z, alpha=0.75, cmap=ListedColormap(('blue', 'red')))
    plt.colorbar(label='Output')

    # Pontos de treino
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolor='k', s=100, cmap=ListedColormap(('blue', 'red')))
    plt.title('Separação Não Linear da MLP')
    plt.xlabel('Entrada 1 (Proteína)')
    plt.ylabel('Entrada 2 (Açúcar)')

    plt.tight_layout()
    plt.show()

# Parâmetros globais do modelo
INPUT_SIZE = 4       # Tamanho da camada de entrada (4 atributos nutricionais)
HIDDEN_SIZE = 4      # Tamanho da camada oculta
OUTPUT_SIZE = 1      # Tamanho da camada de saída (1 neurônio para classificação saudável/não saudável)
LEARNING_RATE = 0.1  # Taxa de aprendizado
EPOCHS = 10000       # Número de épocas de treinamento

# Definindo a classe da MLP para classificação
class MLP:
    def __init__(self):
        # Inicializando pesos e bias com valores aleatórios
        self.input_weights = np.random.rand(INPUT_SIZE, HIDDEN_SIZE)  # Camada de entrada para camada oculta
        self.hidden_weights = np.random.rand(HIDDEN_SIZE, OUTPUT_SIZE) # Camada oculta para saída
        self.input_bias = np.random.rand(1, HIDDEN_SIZE)  # Bias para a camada oculta
        self.hidden_bias = np.random.rand(1, OUTPUT_SIZE)  # Bias para a camada de saída

    def feedforward(self, X):
        # Cálculo da ativação na camada oculta
        self.hidden_layer_activation = np.dot(X, self.input_weights) + self.input_bias
        self.hidden_layer_output = sigmoid(self.hidden_layer_activation)

        # Cálculo da ativação na camada de saída
        self.output_layer_activation = np.dot(self.hidden_layer_output, self.hidden_weights) + self.hidden_bias
        self.output = sigmoid(self.output_layer_activation)
        return self.output

    def backpropagation(self, X, y):
        # Cálculo do erro na saída e delta da camada de saída
        output_error = y - self.output  # Erro = saída esperada - saída da MLP
        output_delta = output_error * sigmoid_derivative(self.output)  # Delta da saída

        # Cálculo do erro e delta na camada oculta
        hidden_error = output_delta.dot(self.hidden_weights.T)  # Propagando o erro para a camada oculta
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer_output)  # Delta da camada oculta

        # Atualização dos pesos e bias (gradiente descendente)
        self.hidden_weights += self.hidden_layer_output.T.dot(output_delta) * LEARNING_RATE
        self.input_weights += X.T.dot(hidden_delta) * LEARNING_RATE
        self.hidden_bias += np.sum(output_delta, axis=0, keepdims=True) * LEARNING_RATE
        self.input_bias += np.sum(hidden_delta, axis=0, keepdims=True) * LEARNING_RATE

    def train(self, X, y):
        mse_values = []  # Lista para armazenar o MSE em cada época
        for epoch in range(EPOCHS):
            self.feedforward(X)
            self.backpropagation(X, y)
            mse_values.append(mse(y, self.output))  # Armazena o MSE da época
        return mse_values

# Inicializando a MLP e treinando o modelo
mlp = MLP()
mse_values = mlp.train(X, y)

# Gráfico do MSE por época
plot_training_curve(EPOCHS, mse_values)

# Geração da separação de classes
plot_decision_boundary(X, y, mlp)
