# Importando bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt

# Dados do problema XOR (entradas e saídas esperadas)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # Entradas
y = np.array([[0], [1], [1], [0]]) # Saídas esperadas

# Funções de ativação e suas derivadas
def sigmoid(x):
    """Função sigmoide, usada para ativação dos neurônios"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivada da função sigmoide, usada para ajuste dos pesos na retropropagação"""
    return x * (1 - x)

# Função para calcular o erro quadrático médio (MSE)
def mse(y_true, y_pred):
    """Cálculo do erro quadrático médio (MSE)"""
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

    # Criando uma malha de pontos para mostrar a separação não linear
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Feedforward para todos os pontos da malha
    Z = mlp.feedforward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Subplot 2: Separação não linear aprendida pela MLP
    plt.subplot(1, 2, 2)
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 50), cmap="coolwarm", alpha=0.8)
    plt.colorbar(label='Output')
    
    # Dados de treino plotados com cores para 0 e 1
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolor='k', marker='o', s=100, cmap="coolwarm")
    plt.title("Separação Não Linear da MLP (Problema XOR)")
    plt.xlabel('Entrada 1')
    plt.ylabel('Entrada 2')
    plt.grid(True)
    plt.show()

# Parâmetros globais do modelo
INPUT_SIZE = 2       # Tamanho da camada de entrada (XOR tem 2 entradas)
HIDDEN_SIZE = 2      # Tamanho da camada oculta (2 neurônios ocultos)
OUTPUT_SIZE = 1      # Tamanho da camada de saída (1 neurônio)
LEARNING_RATE = 0.1  # Taxa de aprendizado
EPOCHS = 10000       # Número de épocas de treinamento

# Definindo a classe da MLP para o problema XOR
class MLP:
    def __init__(self):
        # Inicializando pesos e bias com valores aleatórios
        self.input_weights = np.random.rand(INPUT_SIZE, HIDDEN_SIZE)  # Camada de entrada para camada oculta
        self.hidden_weights = np.random.rand(HIDDEN_SIZE, OUTPUT_SIZE) # Camada oculta para saída
        self.input_bias = np.random.rand(1, HIDDEN_SIZE)  # Bias para a camada oculta
        self.hidden_bias = np.random.rand(1, OUTPUT_SIZE)  # Bias para a camada de saída

    def feedforward(self, X):
        """
        Realiza o processo de feedforward: recebe a entrada X,
        calcula a ativação na camada oculta e na camada de saída.
        """
        # Cálculo da ativação na camada oculta
        self.hidden_layer_activation = np.dot(X, self.input_weights) + self.input_bias
        self.hidden_layer_output = sigmoid(self.hidden_layer_activation)

        # Cálculo da ativação na camada de saída
        self.output_layer_activation = np.dot(self.hidden_layer_output, self.hidden_weights) + self.hidden_bias
        self.output = sigmoid(self.output_layer_activation)
        return self.output

    def backpropagation(self, X, y):
        """
        Realiza o processo de retropropagação: ajusta os pesos e bias
        baseando-se no erro da saída comparado com o valor esperado y.
        """
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
        """
        Função de treinamento: executa feedforward e backpropagation
        para ajustar os pesos e reduzir o erro ao longo das épocas.
        """
        mse_values = []  # Lista para armazenar o MSE em cada época
        for epoch in range(EPOCHS):
            # Executa feedforward e backpropagation
            self.feedforward(X)
            self.backpropagation(X, y)
            # Armazena o erro quadrático médio da época
            mse_values.append(mse(y, self.output))
        return mse_values

# Inicializando a MLP e treinando o modelo
mlp = MLP()
mse_values = mlp.train(X, y)

# Gerando a saída final após o treinamento
predicao = mlp.feedforward(X)
predicao_formatada = np.round(predicao)

# Gráfico do MSE por época
plot_training_curve(EPOCHS, mse_values)

# Resultados
print('\n--- Resultados ---')
print(f'Predições (raw) -> {np.around(predicao, 4).T}')
print(f'Predições (round) -> {predicao_formatada.T}')
print(f'Esperado -> {y.T}')
print(f'Acurácia -> {np.mean(predicao_formatada == y) * 100:.2f}%')
