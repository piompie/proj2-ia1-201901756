# Manual Técnico

Este documento proporciona instrucciones detalladas para la instalación, configuración y uso de una aplicación de machine learning que utiliza TytusJS como base para la implementación de modelos de aprendizaje. La aplicación permite realizar entrenamiento, pruebas y visualización de modelos de manera interactiva mediante una interfaz gráfica.

---

## Tabla de Contenidos
- [Requisitos](#requisitos)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Manipulación del DOM](#manipulación-del-dom)
- [Documentación Técnica de Modelos](#documentación-técnica-de-modelos)
  - [Regresión Lineal y Polinomial](#regresión-lineal-y-polinomial)
  - [Árbol de Decisión](#árbol-de-decisión)
  - [Naïve Bayes](#naïve-bayes)
  - [Red Neuronal](#red-neuronal)
  - [K-means y K-nearest Neighbor](#k-means-y-k-nearest-neighbor)

---

## Requisitos

- **Navegador web**: La aplicación funciona en cualquier navegador moderno que soporte JavaScript.
- **Conexión a Internet**: Se necesitan algunas bibliotecas alojadas en línea.

### Librerías Utilizadas
- **[TytusJS](https://github.com/tytusdb/tytusjs)**: Librería de machine learning para trabajar con modelos como regresión, árboles de decisión, redes neuronales, entre otros.
- **[Chart.js](https://www.chartjs.org/)**: Utilizado para la visualización de datos en gráficos.
- **[vis-network](https://visjs.github.io/vis-network/docs/network/)**: Herramienta para visualizar grafos y redes, útil para árboles de decisión.
- **[GaugeChart](https://github.com/greetclock/gauge-chart)**: Se utiliza para gráficos de indicadores circulares en la visualización de resultados de Naïve Bayes.

---

## Estructura del Proyecto

El proyecto principal consiste en un archivo HTML (`index.html`) que incluye los siguientes elementos:
1. **Formulario de carga de datos**: Permite seleccionar un archivo CSV con los datos para entrenamiento y pruebas.
2. **Selector de modelos**: Un menú desplegable que permite elegir el modelo de machine learning deseado.
3. **Botones de acción**: Incluye botones para ejecutar las funciones de entrenamiento, predicción y generación de gráficos.
4. **Sección de resultados**: Muestra tablas y gráficos generados en función del modelo seleccionado y las acciones realizadas.

### Archivos y Carpetas

- `index.html`: Archivo principal que contiene el código HTML, CSS y JavaScript necesarios.
- **Dependencias en línea**: Las bibliotecas Chart.js, vis-network y GaugeChart se cargan desde CDN externos.

---

## Manipulación del DOM

La aplicación realiza una manipulación dinámica del DOM para construir una interfaz interactiva que permite cargar datos, seleccionar modelos de machine learning, entrenar, predecir, y visualizar resultados.

### Carga de Datos y Generación de Tabla

1. **Carga de Datos**: La aplicación escucha el evento `change` en el input de tipo archivo para cargar el archivo CSV seleccionado por el usuario. Una vez cargado, el archivo se procesa, y sus datos se convierten en una tabla HTML. Esta tabla se inserta en el DOM para que el usuario pueda ver los datos cargados.
2. **Generación de Tabla**: Una función dinámica genera las filas y columnas necesarias de acuerdo a las columnas y valores en el CSV. La tabla se limpia y se reinician las configuraciones previas cada vez que se carga un nuevo archivo.

### Selección de Modelos

El menú desplegable que permite seleccionar el modelo de machine learning escucha el evento `input` para ejecutar una función que carga los elementos de configuración específicos para cada modelo. 
- La manipulación del DOM en este caso permite añadir o actualizar los elementos de configuración (inputs, selects o sliders) en función del modelo seleccionado.

### Configuración de Parámetros del Modelo

Cada modelo tiene configuraciones específicas que se muestran en el DOM de forma dinámica según los requisitos del modelo. Una función central genera los selects y labels adecuados para las variables de entrada, salida, y otros parámetros específicos del modelo.
- **Regresión**: Configura las variables de entrenamiento (`xTrain` y `yTrain`) y prueba (`xTest`), y, en el caso de la regresión polinomial, el grado del polinomio.
- **Árbol de Decisión y Naïve Bayes**: Permiten seleccionar la variable objetivo (columna a predecir) y las características o causas que afectarán a la predicción.
- **Red Neuronal**: Configura el número de capas ocultas y neuronas por capa, además de los inputs y outputs según las columnas seleccionadas.
- **K-means y K-nearest Neighbor**: Permiten seleccionar el número de dimensiones y los datos de entrada en función del modelo de clustering o de clasificación.

### Manejo de Eventos y Botones de Acción

Los botones de acción (**Entrenar**, **Ejecutar** y **Generar Gráficas**) permiten realizar las operaciones del modelo seleccionado. Estos botones escuchan eventos de clic y ejecutan las funciones correspondientes a la operación solicitada:
- **Entrenar**: Ejecuta el método `train` del modelo seleccionado, y actualiza la interfaz para permitir la predicción.
- **Ejecutar**: Ejecuta la función `test`, que genera y muestra las predicciones basadas en los datos de prueba.
- **Generar Gráficas**: Llama al método `graph` para visualizar los resultados de la predicción o del modelo en sí.

### Visualización de Resultados

1. **Resultados de Entrenamiento y Predicción**: Los resultados de entrenamiento y predicción se generan en tablas HTML o en gráficos específicos, y se insertan en divisiones `<div>` adecuadas del DOM.
2. **Gráficos**: La visualización de gráficas se implementa utilizando bibliotecas como Chart.js o GaugeChart, según el tipo de gráfico requerido. Estos gráficos se insertan en el DOM dentro de un `<canvas>`, y se actualizan dinámicamente en función del modelo seleccionado.

Esta estructura de manipulación del DOM permite que la aplicación sea dinámica e interactiva, proporcionando una experiencia fluida para el usuario al configurar y operar modelos de machine learning de manera visual.

---

## Documentación Técnica de Modelos

### Regresión Lineal y Polinomial
La regresión permite predecir valores continuos basándose en variables independientes.

#### Configuración
- **Variables de entrenamiento**: `xTrain` y `yTrain`.
- **Variables de prueba**: `xTest`.
- **Parámetros adicionales**: En la regresión polinomial, puedes especificar el grado del polinomio.

#### Implementación
- **train()**: Se ajusta el modelo a los datos de entrenamiento.
- **test()**: Calcula las predicciones para `xTest`.
- **graph()**: Grafica los puntos de datos y la línea de regresión o curva polinomial.

### Árbol de Decisión
Este modelo clasifica datos mediante un árbol de decisión basado en el algoritmo ID3.

#### Configuración
- **Variable objetivo**: La columna a predecir.
- **Características**: Las columnas de entrada que influyen en la predicción.

#### Implementación
- **train()**: Genera el árbol de decisión a partir de los datos de entrenamiento.
- **test()**: Clasifica una muestra de prueba.
- **graph()**: Visualiza el árbol con `vis-network` para mostrar las decisiones en nodos y ramas.

### Naïve Bayes
Método de clasificación probabilística que calcula la probabilidad de que una instancia pertenezca a una clase.

#### Configuración
- **Causas**: Selección de variables de entrada.
- **Efecto**: La variable objetivo que se desea predecir.

#### Implementación
- **train()**: Define las relaciones causa-efecto.
- **test()**: Realiza la predicción y muestra la probabilidad de cada clase.
- **graph()**: Muestra los resultados con un gráfico de GaugeChart.

### Red Neuronal
Red neuronal básica con capas configurables, adecuada para predicciones clasificatorias o de regresión.

#### Configuración
- **Número de entradas y salidas**: Selección de columnas de entrada y salida.
- **Capas ocultas**: Configuración del número de capas ocultas y neuronas por capa.

#### Implementación
- **train()**: Entrena la red con datos configurados en múltiples iteraciones.
- **test()**: Realiza la predicción y muestra la probabilidad de cada clase.
- **graph()**: Grafica un gráfico de dona que muestra la probabilidad de cada clase.

### K-means y K-nearest Neighbor
**K-means** agrupa los datos en clusters mientras que **K-nearest Neighbor** clasifica puntos en función de sus vecinos más cercanos.

#### K-means
- **train()**: Agrupa los datos en k clusters usando el algoritmo de clustering.
- **test()**: Clasifica puntos en el espacio de acuerdo a sus centros de cluster.
- **graph()**: Muestra un gráfico de dispersión con los clusters y centroides en colores.

#### K-nearest Neighbor
- **train()**: Configura los datos y etiquetas de grupo.
- **test()**: Clasifica una muestra en función de sus vecinos más cercanos.
- **graph()**: Muestra una tabla con las distancias y etiquetas de los vecinos.

## Clases Utilizadas para los Modelos

La aplicación usa clases específicas de TytusJS para implementar los distintos modelos de machine learning. A continuación, se describen las clases utilizadas, sus métodos principales y cómo contribuyen a la funcionalidad de cada modelo.

---

### Clase `LinearRegression` - Regresión Lineal

La clase `LinearRegression` permite realizar regresiones lineales simples, ajustando una línea que mejor representa la relación entre dos variables.

#### Métodos Principales
- **fit(xTrain, yTrain)**: Ajusta el modelo usando los datos de entrenamiento `xTrain` y `yTrain`.
- **predict(xTest)**: Realiza predicciones para los valores en `xTest` en base al modelo ajustado.
- **getCoefficients()**: Devuelve los coeficientes de la regresión lineal (pendiente e intercepto).

---

### Clase `PolynomialRegression` - Regresión Polinomial

La clase `PolynomialRegression` permite realizar regresiones polinomiales para modelos de mayor complejidad, ajustando curvas en lugar de líneas.

#### Métodos Principales
- **fit(xTrain, yTrain, degree)**: Ajusta el modelo usando los datos de entrenamiento y el grado `degree` del polinomio.
- **predict(xTest)**: Calcula predicciones para el conjunto `xTest`.
- **getCoefficients()**: Devuelve los coeficientes del polinomio ajustado.

---

### Clase `DecisionTreeID3` - Árbol de Decisión

Esta clase implementa el algoritmo ID3 para generar árboles de decisión, usado en tareas de clasificación.

#### Métodos Principales
- **train(dataSet)**: Entrena el árbol de decisión con un conjunto de datos `dataSet`.
- **predict(sample)**: Clasifica una muestra `sample` en función del árbol de decisión entrenado.
- **generateDotString(root)**: Genera una representación gráfica en formato DOT para visualizar el árbol.

---

### Clase `NaiveBayes` - Clasificación Naïve Bayes

La clase `NaiveBayes` permite realizar clasificación probabilística mediante el cálculo de la probabilidad de que una instancia pertenezca a una clase, basada en características observadas.

#### Métodos Principales
- **insertCause(name, values)**: Define una causa en el modelo Naïve Bayes, asociando un conjunto de valores a una variable `name`.
- **predict(effect, causes)**: Calcula la probabilidad de una variable objetivo `effect` dada una lista de causas observadas.
- **getProbabilities()**: Devuelve un objeto con las probabilidades de cada clase.

---

### Clase `NeuralNetwork` - Red Neuronal

La clase `NeuralNetwork` permite configurar y entrenar una red neuronal básica con un número personalizado de capas y neuronas.

#### Métodos Principales
- **constructor(layers)**: Define la estructura de la red usando un array `layers` que especifica el número de neuronas en cada capa.
- **Entrenar(inputs, outputs)**: Realiza el entrenamiento de la red ajustando los pesos en función de los `inputs` y `outputs`.
- **Predecir(inputs)**: Genera una predicción basada en los `inputs` dados.

---

### Clases `LinearKMeans` y `_2DKMeans` - Clustering K-means

Estas clases implementan el algoritmo de clustering K-means, que agrupa los datos en `k` clusters. `LinearKMeans` se usa para una sola dimensión y `_2DKMeans` para datos bidimensionales.

#### Métodos Principales
- **clusterize(k, data, iterations)**: Agrupa los datos `data` en `k` clusters usando un número específico de iteraciones `iterations`.
- **getCentroids()**: Devuelve los centroides de los clusters resultantes.
- **assignClusters()**: Asigna cada punto al cluster más cercano.

---

### Clase `KNearestNeighbor` - K-nearest Neighbor

La clase `KNearestNeighbor` permite clasificar puntos en función de sus vecinos más cercanos, usando distancias euclidiana y de Manhattan.

#### Métodos Principales
- **inicializar(k, data, labels)**: Configura el modelo con un número `k` de vecinos, un conjunto de datos `data` y etiquetas `labels`.
- **predecir(point)**: Clasifica un `point` en función de los vecinos más cercanos.
- **euclidean(point)** y **manhattan(point)**: Calcula las distancias euclidiana y de Manhattan de `point` respecto a otros puntos.
