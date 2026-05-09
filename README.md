# Estructura del proyecto

Este documento describe la organización inicial del código del TFG y la función de cada archivo.  
El objetivo es mantener una estructura clara para poder ampliar el modelo de forma progresiva.

## Descripción general

El proyecto implementa un modelo multinivel de cooperación en poblaciones heterogéneas.  
En la versión actual se ha implementado principalmente la dinámica interna de las comunidades.

Cada comunidad se modela como una red de individuos con topología *small-world*. Los individuos interactúan con sus vecinos mediante un dilema del prisionero débil y actualizan su estrategia a partir de una combinación de factores sociales y una dinámica evolutiva tipo Fermi.

Actualmente, el modelo permite:

- crear varias comunidades con tamaños distintos;
- inicializar una proporción aleatoria de cooperadores en cada comunidad;
- generar redes internas tipo *small-world*;
- calcular payoffs individuales acumulados;
- calcular componentes sociales de decisión;
- aplicar una probabilidad de revisión de estrategia;
- aplicar una regla tipo Fermi en caso de revisión;
- evolucionar cada comunidad durante un número fijo de rondas;
- obtener métricas agregadas por comunidad.

## Archivos principales

### `main.py`

Archivo principal de ejecución.

Se utiliza para lanzar pruebas rápidas del modelo.  
Actualmente permite crear una configuración, inicializar el modelo con varias comunidades, mostrar el estado inicial, ejecutar la dinámica interna y mostrar el estado final.

Este archivo no contiene la lógica del modelo, sino que sirve como punto de entrada para ejecutar simulaciones sencillas.

---

### `src/config.py`

Contiene la configuración general del modelo.

Aquí se definen los principales parámetros utilizados en la simulación, como:

- número de comunidades;
- tamaño mínimo y máximo de las comunidades;
- parámetros de la red *small-world*;
- parámetro de tentación del dilema del prisionero débil;
- intervalo inicial de cooperadores;
- longitud de memoria;
- número de rondas internas;
- pesos de la propensión social;
- error de ejecución;
- parámetros de la dinámica Fermi;
- parámetros del nivel intercomunitario.

La idea es centralizar los parámetros en este archivo para evitar escribir valores fijos directamente en el resto del código.

---

### `src/networks.py`

Contiene funciones relacionadas con la creación de redes.

En la versión actual incluye la función para crear redes internas tipo *small-world* mediante el modelo de Watts-Strogatz.

Más adelante, este archivo también podrá incluir la construcción de la red intercomunitaria, donde cada nodo representará una comunidad.

---

### `src/community.py`

Contiene la clase `Community`.

Este es el archivo principal de la dinámica intracomunitaria.  
Cada objeto de tipo `Community` representa una comunidad formada por individuos conectados en una red.

Actualmente implementa:

- creación de la red interna;
- inicialización aleatoria de cooperadores;
- cálculo del payoff individual;
- almacenamiento del historial de estrategias;
- cálculo de la memoria temporal;
- cálculo de componentes sociales:
  - estado interno;
  - reciprocidad directa;
  - reputación;
  - entorno local;
- cálculo de la propensión social a cooperar;
- normalización de payoffs;
- cálculo de la probabilidad de revisión de estrategia;
- regla de comparación tipo Fermi;
- actualización síncrona de estrategias;
- error de ejecución;
- evolución durante varias rondas;
- métricas agregadas de comunidad.

La dinámica actual distingue entre comportamiento social habitual y revisión evolutiva.  
El individuo no aplica Fermi en cada ronda de forma automática, sino solo cuando entra en revisión de estrategia.

---

### `src/model.py`

Contiene la clase `MultilevelCooperationModel`.

Este archivo coordina el funcionamiento de varias comunidades.  
Actualmente permite:

- crear un conjunto de comunidades;
- asignar tamaños aleatorios dentro del intervalo definido;
- ejecutar la dinámica interna de todas las comunidades;
- obtener un resumen agregado de cada comunidad.

Más adelante, este archivo incorporará también el nivel intercomunitario del modelo, incluyendo:

- construcción de la red entre comunidades;
- cálculo de probabilidades de cooperación externa;
- interacciones entre comunidades;
- cálculo de la cooperación global final.

---

### `src/metrics.py`

Archivo reservado para métricas adicionales.

En la versión actual puede estar vacío o contener funciones auxiliares.  
Más adelante podrá utilizarse para calcular indicadores como:

- cooperación media global;
- evolución temporal de la cooperación;
- payoff medio global;
- varianza entre comunidades;
- diferencias entre comunidades grandes y pequeñas;
- métricas de estabilidad del sistema.

---

### `src/plots.py`

Archivo reservado para visualización de resultados.

Más adelante podrá utilizarse para generar gráficas como:

- evolución de la cooperación por ronda;
- comparación entre comunidades;
- cooperación final según parámetros;
- relación entre tamaño de comunidad y cooperación;
- efecto de la intensidad de selección Fermi;
- efecto de la probabilidad de revisión.

---

### `experiments/run_baseline.py`

Archivo reservado para experimentos más organizados.

La idea es usarlo cuando el modelo base ya esté funcionando correctamente y se quieran ejecutar simulaciones de forma sistemática.

Por ejemplo:

- simulación base;
- comparación de distintos valores de `b`;
- comparación de distintos valores de `lambda_fermi`;
- análisis de sensibilidad de los pesos sociales;
- comparación entre distintas topologías;
- varias repeticiones con distintas semillas.

---

### `results/data/`

Carpeta reservada para guardar resultados numéricos.

Podrá contener archivos `.csv`, `.json` o similares con los resultados de las simulaciones.

Ejemplos:

- cooperación final por comunidad;
- evolución temporal de la cooperación;
- parámetros utilizados en cada experimento;
- resultados agregados de varias simulaciones.

---

### `results/figures/`

Carpeta reservada para guardar figuras.

Podrá contener gráficas generadas durante el análisis del modelo.

Ejemplos:

- evolución temporal de la cooperación;
- comparación entre escenarios;
- gráficos de sensibilidad;
- visualizaciones de redes.

---

### `requirements.txt`

Archivo con las librerías necesarias para ejecutar el proyecto.

En la versión actual incluye principalmente:

- `numpy`;
- `pandas`;
- `networkx`;
- `matplotlib`.

Estas librerías permiten trabajar con arrays, tablas de resultados, redes y visualización.

## Estado actual del modelo

La versión actual del código implementa la dinámica interna de varias comunidades y permite obtener resultados agregados antes y después de la evolución.

Los primeros resultados muestran que, con la parametrización inicial utilizada, la cooperación cae de forma notable tras la evolución interna de las comunidades.

Esto no debe interpretarse todavía como un resultado definitivo del modelo, ya que:

- los parámetros se han fijado de forma orientativa;
- no se ha realizado aún calibración;
- el modelo todavía está en una versión inicial;
- no se ha incorporado completamente el nivel intercomunitario;
- no se han realizado experimentos de sensibilidad;
- no se han comparado diferentes configuraciones.

Por tanto, los resultados actuales son útiles principalmente para verificar que la implementación funciona correctamente y que el modelo produce una dinámica interna coherente.  
El análisis matemático y experimental deberá realizarse en fases posteriores, ajustando parámetros y comparando escenarios.

## Próximos pasos

Los siguientes pasos previstos son:

1. Guardar métricas temporales de la evolución de cada comunidad.
2. Crear gráficas de cooperación por ronda.
3. Revisar la sensibilidad del modelo a los parámetros principales.
4. Implementar la red intercomunitaria.
5. Calcular la cooperación externa de cada comunidad.
6. Obtener una medida global de cooperación del sistema.
7. Diseñar experimentos sistemáticos para analizar los resultados.