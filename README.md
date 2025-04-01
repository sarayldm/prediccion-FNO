# Predicción de la Fiebre del Virus del Nilo Occidental con Machine Learning
## Descripción 
Este proyecto aplica **machine learning supervisado** para predecir la presencia de la **fiebre del virus del Nilo Occidental (FNO)** en distintas provincias de España. Se utiliza un modelo de **clasificación binaria**, donde la salida indica si hay **presencia (1) o ausencia de casos (0)** en una determinada ubicación y fecha.  

## Fuente de datos
Los datos utilizados en este proyecto provienen de distintas fuentes oficiales y bases de datos abiertas.

Las principales variables analizadas son: 

🌤️**Variables climáticas**
- Temperatura media (ºC)
- Humedad relativa (%)
- Precipitación (mm)
- Velocidad del viente (km/h)
Obtenido de [Meteostat](https://meteostat.net/es/) y [Open-Meteo](https://open-meteo.com/)

🏙️**Variables poblacionales**
- Censo de la población
Obtenido de [Istat](https://www.istat.it/) y [INE](https://www.ine.es/)

🌍**Variables geoespaciales**
- Latitud del centro geograáfico de la provincia.
- Latitud del centro geográfico de la provincia.
Obtenido de [GeoPy](https://geopy.readthedocs.io/en/stable/)

## Evaluación del modelo
El rendimiento del modelo se evalúa mediante:
- **Metricas de clasificación**: accuracy, recall, precision y F1-score
- **Curvas AUC-ROC**
- **Importancia de las características** basada en SHAP





 
