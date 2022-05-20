# New_york_trip_duration

## Notebook de entrenamiento 

El notebook <code>Exploratory_Data.ipynb </code> contiene análisis exploratioro de los datos, procesamiento, selección de variables y entrenamiento de modelos baselines. Asimismo, contiene anotaciones sobre cada paso ejecutado

## Reporte de Pandas profilling

* Se tiene mayor cantidad de registros del vendor 2
* Se tiene mayor cantidad de viajes con un solo pasajero (70.9% de los viajes)
* En la variables "store_and_fwd_flag" predomina el False
* La hora de recojo tiene mayor acumulación en horas más tardes
* La variable de trip duration tiene más correlación con pickup longitud y dropoff longitud

## Ejecución de archivos .py

Se ha desarrollado estos archivos para una facilidad en la optimización de hiperparámetros y futuro reentrenamiento(entrenamiento.py). Mientras que el archivo prediction.py sirve para ejecutar el modelo a demanda

## Estructura de carpeta 'outputs'

* **hyperparameter_tuning:** Se guarda un archivo sqlite donde se almacena todas las combinaciones de hiperparámetros realizadas utilizando la optimización bayesina. Contiene parámetros y resultados.
* **imp_variables** Se guarda la importancia de variables para el modelo
* **models** se guardan los modelos seleccionados como mejores en cada entrenamiento
* **preds** se guardan los archivos outputs (con la predicción)
* **standarizers** se guarda archivos pickles de las estandarizaciones que se han trabajado

## Next steps

* Excluir outliers para el entrenamiento: tales como viajes de 0 pasajeros o más de 4. Asi como puntos de recojo que estén en lugares inhóspitos tales como el mar (se visualiza mejor en el power bi)
* Explorar más variables como input al modelo
* Adaptación a un sistema para que el modelo puede ejecutarse en tiempo real

## Principales insights

El objetivo de este modelo es predecir el tiempo de duración del viaje para conocimiento del conductor y pasajero. Al mismo tiempo, a partir de esto se puede pasar a una segunda fase de optimización de tiempos.
