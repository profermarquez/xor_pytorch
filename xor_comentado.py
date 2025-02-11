# Primero importamos las librerias necesarias:

#import torch#: Importa la biblioteca principal de PyTorch, que es una biblioteca de aprendizaje automático.

#from torch.autograd import Variable#: Importa la clase Variable, que se utilizaba para crear variables autograd en versiones anteriores de PyTorch (ahora se usa directamente torch.Tensor).

#import torch.nn as nn:#Importa el módulo torch.nn que contiene clases y métodos para construir redes neuronales.

#import torch.nn.functional# as F: Importa el módulo torch.nn.functional, que proporciona funciones útiles para crear y manipular capas de redes neuronales.

#import torch.optim as optim#: Importa el módulo torch.optim, que contiene optimizadores para ajustar los pesos de la red neuronal.

#import numpy as np: Importa NumPy#, una biblioteca para operaciones numéricas y manejo de arrays.

#import matplotlib.pyplot as plt#: Importa Matplotlib, una biblioteca para la visualización de datos.

#%matplotlib inline#: Línea mágica de Jupyter Notebook, que en este caso se utiliza en colab para mostrar las gráficas de Matplotlib dentro del notebook.


#torch.manual_seed(2):#Establece una semilla para los generadores de números aleatorios de PyTorch, asegurando que los resultados sean reproducibles.

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(2)

X = torch.Tensor([[0,0],[0,1], [1,0], [1,1]])
Y = torch.Tensor([0,1,1,0]).view(-1,1)

#El código define una clase XOR que crea una red neuronal simple para resolver el problema del XOR. Aquí está el desglose de las funciones dentro de la clase:

# __init__(self, input_dim=2, output_dim=1)
# Este es el método constructor que se ejecuta cuando se crea una instancia de la clase XOR.

#super(XOR, self).__init__():# Llama al constructor de la clase base nn.Module para inicializar correctamente la red neuronal.
#lin1 = nn.Linear(input_dim, 2) Define la primera capa lineal (completamente conectada) que toma input_dim entradas (2 en este caso) y produce 2 salidas.
#lin2 = nn.Linear(2, output_dim)  Define la segunda capa lineal que toma 2 entradas (de la capa anterior) y produce output_dim salidas (1 en este caso).
# forward(self, x)

#Este método define cómo se pasa la información a través de la red.

#x = self.lin1(x): Aplica la primera capa lineal a la entrada x.
#x = F.sigmoid(x): Aplica la función de activación sigmoide a la salida de la primera capa.
#x = self.lin2(x): Aplica la segunda capa lineal a la salida de la función sigmoide.
# return x: Devuelve la salida final de la red.

class XOR(nn.Module):
    def __init__(self, input_dim = 2, output_dim=1):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 2)
        self.lin2 = nn.Linear(2, output_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = F.sigmoid(x)
        x = self.lin2(x)
        return x

# Primero crea una instancia de la clase XOR, es decir, una red neuronal con la estructura definida anteriormente.

# Luego inicialización de Pesos: m.weight.data.normal_(0, 1): Inicializa los pesos de la capa lineal con una distribución normal (media 0, desviación estándar 1)

# Por ultimo:
#loss_func = nn.MSELoss():# Define la función de pérdida como el error cuadrático medio (MSE), que mide la diferencia entre las predicciones del modelo y los valores reales.

#optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)#: Define el optimizador como Gradiente Descendente Estocástico (SGD) con una tasa de aprendizaje de 0.02 y un momento de 0.9. El optimizador se encargará de actualizar los parámetros del modelo durante el entrenamiento para minimizar la función de pérdida.

model = XOR()
def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 1)

weights_init(model)
loss_func = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

#Nota al margen#: El Gradiente Descendente Estocástico (SGD) es un método de optimización utilizado para entrenar modelos de aprendizaje automático. En lugar de calcular el gradiente de la función de pérdida usando todo el conjunto de datos (como en el gradiente descendente estándar), SGD utiliza un solo ejemplo de entrenamiento (o un pequeño lote) para actualizar los parámetros del modelo en cada iteración. Esto hace que el proceso sea más rápido y eficiente, especialmente para grandes conjuntos de datos, aunque introduce más ruido en las actualizaciones de los parámetros.

# Luego procedemos a ejecutar el #bucle de entrenamiento# por lotes con descenso de gradiente estocástico para un modelo de red neuronal. Itera a través de los datos de entrenamiento, calcula la pérdida para cada punto de datos, realiza la propagación hacia atrás para actualizar los parámetros del modelo y registra periódicamente el progreso del entrenamiento.

#Primero Iniciamos:

# epochs: Establece el número de iteraciones de entrenamiento (2001 en este caso).
# steps: Establece el número de iteraciones por época, probablemente igual al número de puntos de datos en X (asumiendo que X son sus datos de entrenamiento).

# Segundo entrenamos:

#El bucle for externo itera epochs veces, representando el proceso de entrenamiento general.
# El bucle for interno itera a través de cada punto de datos en X una vez por época (se puede agregar aleatoriedad para un mejor entrenamiento).

# Tercero seleccionamos los datos:

# data_point = np.random.randint(X.size(0)): Selecciona aleatoriamente un índice de 0 al tamaño de X (número de puntos de datos). Esto asegura que se utilicen diferentes puntos de datos en cada iteración para el descenso del gradiente estocástico.

# Cuarto creamos algunas variables:

# x_var = Variable(X[data_point], requires_grad=False): Crea una Variable (probablemente un tensor de la biblioteca de aprendizaje profundo) que contiene el punto de datos seleccionado de X. requires_grad=False indica que no necesitamos calcular gradientes para los datos de entrada en sí.
# y_var = Variable(Y[data_point], requires_grad=False): De manera similar, crea una Variable para el valor objetivo correspondiente de Y.

# Quinto restablecemos el gradiente:

#optimizer.zero_grad(): Restablece los gradientes acumulados en el optimizador de iteraciones anteriores. Esto es crucial antes de la retropropagación para cada punto de datos.

# Sexto cálculamos la predicción y pérdida:

# y_hat = model(x_var): Pasa los datos de entrada (x_var) a través del modelo de red neuronal (model) para obtener la salida predicha (y_hat).
# loss = loss_func.forward(y_hat, y_var): Calcula la pérdida (error) entre la salida predicha (y_hat) y el valor objetivo (y_var) utilizando la función de pérdida especificada (loss_func.forward).


# Y lor ultimo realizamos la retropropagación:

# loss.backward(): Realiza la propagación hacia atrás, un proceso que propaga la pérdida calculada hacia atrás a través de la red, calculando los gradientes para cada peso

#Nota: #La propagación hacia atrás es un proceso crucial que permite a la red neuronal aprender de sus errores y mejorar su rendimiento de forma iterativa.

epochs = 3000
steps = X.size(0)
for i in range(epochs):
    for j in range(steps):
        data_point = np.random.randint(X.size(0))
        x_var = Variable(X[data_point], requires_grad=False)
        y_var = Variable(Y[data_point], requires_grad=False)

        optimizer.zero_grad()
        y_hat = model(x_var)
        loss = loss_func.forward(y_hat, y_var)
        loss.backward()
        optimizer.step()

    if i % 500 == 0:
        print (i, loss.data)

# Finalmente graficamos para observar las fronteras de decisión aprendidas por las dos primeras neuronas en la primera capa oculta por el modelo de red neuronal entrenado.



model_params = list(model.parameters())

model_weights = model_params[0].data.numpy()
model_bias = model_params[1].data.numpy()

plt.scatter(X.numpy()[[0,-1], 0], X.numpy()[[0, -1], 1], s=50)
plt.scatter(X.numpy()[[1,2], 0], X.numpy()[[1, 2], 1], c='red', s=50)

x_1 = np.arange(-0.1, 1.1, 0.1)
y_1 = ((x_1 * model_weights[0,0]) + model_bias[0]) / (-model_weights[0,1])
plt.plot(x_1, y_1)

x_2 = np.arange(-0.1, 1.1, 0.1)
y_2 = ((x_2 * model_weights[1,0]) + model_bias[1]) / (-model_weights[1,1])
plt.plot(x_2, y_2)
plt.legend(["neuron_1", "neuron_2"], loc=8)
plt.show()

# Evaluamos las predicciones de la compuerta Xor

print("Imprimimos algunos casos según el modelo entrenado:")
print("0 xor 0 => ", round( model(torch.tensor([0.0, 0.0])).item() ))
print("0 xor 1 => ", round( model(torch.tensor([0.0, 1.0])).item() ))
print("1 xor 0 => ", round( model(torch.tensor([1.0, 0.0])).item() ))
print("1 xor 1 => ", round( model(torch.tensor([1.0, 1.0])).item() ))