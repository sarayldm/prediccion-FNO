# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 12:16:29 2025

@author: saray
"""

# Bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve 
from imblearn.pipeline import Pipeline 
from imblearn.over_sampling import SMOTE
import joblib
import os

# ----------------------------
# 1. FUNCION DE CARGA Y PREPROCESAMIENTO
# ----------------------------
def load_and_preprocess(filepath):
    try:
        df = pd.read_csv(filepath, sep=';')

        # Crear target
        df['outbreak'] = df['new_cases'].apply(lambda x: 1 if x > 0 else 0)

        # Convertir la columna de fecha
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')

        # Crear columnas temporales adicionales
        df['year'] = df['data'].dt.year
        df['month'] = df['data'].dt.month
        
        # Características estacionales mejoradas
        df['month_sin'] = np.sin(2 * np.pi * df['month']/ 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/ 12)
        
        # Eliminar columnas no numéricas (después de ingeniería de características)
        df.drop(['data', 'region', 'province', 'new_cases', 'month'], axis=1, inplace=True, errors='ignore')

        return df.dropna().reset_index(drop=True)
    
    except Exception as e:
        print(f"Error en la carga y preprocesamiento: {str(e)}")
        raise
        
# ----------------------------
# 2. DEFINICION DE MODELOS Y PARAMETROS (POR DEFECTO)
# ----------------------------
models = {
    'RandomForest': RandomForestClassifier(class_weight='balanced', random_state= 42),
    'GradientBoosting': GradientBoostingClassifier(random_state= 42),
    'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=42)
    }

# ----------------------------
# 3. EVALUACION DEL DESEMPEÑO DE LOS MODELOS - ENTRENAMIENTO MODELO (ITALIA)
# ----------------------------
def model_evaluation(X,y, show_feature_importance=False):
    try:
        # Validación cruzada con TimeSeriesSplit (ajustar según las necesidades)
        tscv = TimeSeriesSplit(n_splits=5)
        # Lista para guardar los resultados
        results = []

        for model_name, model in models.items():
            pipeline = Pipeline([
                ('scaler',StandardScaler()),
                ('classifier', model) 
            ])

            # Listas para guardar los valores de las métricas
            metrics = {
                'accuracy' : [],
                'f1' : [],
                'recall' : [],
                'precision' : [],
                'roc_auc' : []
            }

            # Lista para guardar las etiquetas y las probabilidades 
            all_y_test = []
            all_y_proba = []

            # Realizar la validación cruzada
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                #Calcular sample weights solo para GradientBoosting
                if model_name == 'GradientBoosting':
                    sample_weights = compute_sample_weight(class_weight="balanced", y = y_train)
                    pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
                else:
                    #Entrenar el modelo
                    pipeline.fit(X_train, y_train)

                # Predicción de clases (0 o 1) y probabilidad de la clase 1
                y_pred = pipeline.predict(X_test)
                y_proba = pipeline.predict_proba(X_test)[:,1]
                 
                # Guardar métricas
                metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                metrics['f1'].append(f1_score(y_test, y_pred))
                metrics['recall'].append(recall_score(y_test, y_pred))
                metrics['precision'].append(precision_score(y_test, y_pred))
                metrics['roc_auc'].append(roc_auc_score(y_test, y_proba))

                # Guardar las etiquetas y probabilidades
                all_y_test.append(y_test)
                all_y_proba.append(y_proba)

            # Guardar el resultado
            results.append({
                'model': model_name,
                'metrics': {k: np.mean(v) for k, v in metrics.items()},
                'y_test': np.concatenate(all_y_test),
                'y_proba': np.concatenate(all_y_proba),
                'pipeline': pipeline
            })

            # Imprimir el promedio de las metricas principales del modelo evaluado
            print(f"\n{model_name} - Resultados:")
            print(classification_report(y_test, y_pred))
            print(f"Accuracy: {np.mean(metrics['accuracy']):.4f}")
            print(f"F1-Score: {np.mean(metrics['f1']):.4f}")
            print(f"Recall: {np.mean(metrics['recall']):.4f}")
            print(f"Precision: {np.mean(metrics['precision']):.4f}")
            print(f"AUC-ROC: {np.mean(metrics['roc_auc']):.4f}")

            # Mostrar la importancia de las características 
            if show_feature_importance:
                classifier = pipeline.named_steps['classifier']
                
                # Para modelos que tienen 'feature_importances_' (Random Forest, Gradient Boosting)
                if hasattr(classifier, 'feature_importances_'):
                    importances = classifier.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    print(f"\nImportancia de características para {model_name}:")
                    for idx in indices[:10]: # Mostrar solo las 10 más importantes
                        print(f"{X.columns[idx]}: {importances[idx]:.4f}")
                    
                    # Graficar la importancia de las características
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x=importances[indices[:10]], y=X.columns[indices[:10]], palette='viridis')
                    plt.title(f'Importancia de las características - {model_name}', 
                              fontsize=14, fontweight='bold')
                    plt.xlabel('Importancia', fontsize=12)
                    plt.ylabel('Característica', fontsize=12)
                    plt.show()
                
                # Para modelos lineales como Regresión Logística
                elif hasattr(classifier, 'coef_'):
                    coef = classifier.coef_[0]  # Extraer los coeficientes
                    importances = np.abs(coef)  # Tomamos el valor absoluto de los coeficientes
                    indices = np.argsort(importances)[::-1]  # Ordenar por la importancia
                    print(f"\nImportancia de las características para {model_name}:")
                    for idx in indices[:10]:  # Mostrar solo las 10 más importantes
                        print(f"{X.columns[idx]}: {importances[idx]:.4f}")

                    # Graficar la importancia de las características
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x=importances[indices[:10]], y=X.columns[indices[:10]], palette='viridis')
                    plt.title(f'Importancia de las características - {model_name}', 
                              fontsize=16, fontweight='bold')
                    plt.xlabel('Importancia', fontsize=12)
                    plt.ylabel('Característica', fontsize=12)
                    plt.show()
        
        return sorted(results, key=lambda x: x['metrics']['recall'], reverse=True) #Metrica que mas nos interesa

    except Exception as e:
        print(f"Error en el entrenamiento: {str(e)}")
        raise
        
# ----------------------------
# 4. GUARDADO DE RESULTADOS Y GRAFICOS
# ----------------------------            
def save_result(results, best_model, country):
    try:
        # Crear directorio para guardar los resultados
        os.makedirs('results', exist_ok=True)

        # Guardar metricas en CSV
        data = []
        for r in results:
            row = {'Modelo': r['model'], **r['metrics']} # Obtiene el nombre del modelo y desempaqueta las metricas
            data.append(row)

        df_metrics = pd.DataFrame(data)
        df_metrics.to_csv(f'results/metrics_{country}.csv', index=False)

        # Guardar mejor modelo
        joblib.dump(best_model, f'results/best_model_{country}.pkl')
        
        # Guardar scaler
        scaler = best_model.named_steps['scaler']  
        joblib.dump(scaler, f'results/scaler_{country}.pkl')

        # Gráficos comparativos de las metricas principales
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))

        # Metricas a comparar 
        metrics_comparison = ['accuracy', 'f1', 'recall', 'precision', 'roc_auc']
        labels = ["A", "B", "C", "D", "E"]  
      
        model_colors = {
            'LogisticRegression': '#1f77b4',
            'GradientBoosting': '#ff7f0e',
            'RandomForest': '#2ca02c'
            }
        model_name_mapping = {
            "RandomForest": "Random Forest",
            "LogisticRegression": "Logistic Regression",
            "GradientBoosting": "Gradient Boosting"
            }

        for i, (label, metric) in enumerate(zip(labels, metrics_comparison)):
            ax = axs.flatten()[i]
            values = [r['metrics'][metric] for r in results]
            model_names = [model_name_mapping.get(r['model'], r['model']) for r in results]
            ax.bar(model_names, values, color=[model_colors[r['model']] for r in results])
            ax.tick_params(axis='x', labelsize=14)
            
            # Ajuste en el título con la etiqueta
            metric_title = 'AUC-ROC' if metric == 'roc_auc' else metric.capitalize()
            ax.set_title(f"{label}. {metric_title}", fontweight='bold',fontsize=18)
           
            ax.set_ylim(0, 1)
            

        # Curvas ROC
        for idx, r in enumerate(results):
            y_test = r['y_test']
            y_proba = r['y_proba']
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = roc_auc_score(y_test, y_proba)
            model_names = [model_name_mapping.get(r['model'], r['model']) for r in results]
            axs[1,2].plot(fpr, tpr, label=f"{model_names[idx]}")

        axs[1,2].plot([0,1], [0,1], 'k--')
        axs[1,2].set_title('F. Curvas ROC', fontweight='bold', fontsize=18)
        axs[1,2].legend(loc="lower right", fontsize=15)

        plt.tight_layout()
        plt.savefig(f'results/model_comparison_{country}.png')
        plt.close()

    except Exception as e:
        print(f"Error en el guardado: {str(e)}")
        raise

# Gráfico distribución ausencia - presencia   
def plot_outbreak_distribution(df, country):
    counts = df['outbreak'].value_counts().sort_index()
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=counts.index, y=counts.values, palette=['#1f77b4', '#ff7f0e'])
    plt.xticks([0, 1], [r'$\mathbf{0}$' + '\nAusencia de casos', r'$\mathbf{1}$' + '\nPresencia de casos'], ha='center')
    plt.ylabel('Frecuencia')
    plt.xlabel("")
    #plt.title(f'Distribución de la variable "outbreak" - {country}')
    
    for i, v in enumerate(counts.values):
        ax.text(i, v/2, str(v), ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    plt.show()   
    
# ----------------------------
# 5. TRANSFER LEARNING (ESPAÑA)
# ----------------------------
def transfer_learning(best_model, X_es, y_es):
    try:
        classifier = best_model.named_steps['classifier']

        # Aplicar warm_start solo si el modelo lo permite
        if isinstance(classifier, (RandomForestClassifier, GradientBoostingClassifier)): # Verificar si es uno de estos dos modelos
            classifier.warm_start = True
            classifier.n_estimators += int(classifier.n_estimators * 0.2) # Aumenta en un 20%
            
        best_model.fit(X_es, y_es)

        return best_model

    except Exception as e:
        print(f"Error en el transfer learning: {str(e)}")
        raise

# ----------------------------
# 6. EJECUCION PRINCIPAL
# ----------------------------
if __name__ == "__main__":

    ## Entrenamiento con datos de Italia ##

    # Carga y preprocesamiento
    df_italy = load_and_preprocess('italy_data.csv')
    X_it = df_italy.drop('outbreak', axis=1)
    y_it = df_italy['outbreak'] 

    # Evaluacion de los modelos
    model_results = model_evaluation(X_it, y_it, show_feature_importance=True)
    best_model = model_results[0]['pipeline']

    # Guardado de los resultados 
    save_result(model_results, best_model, 'Italia')
    
    # Grafico de distribucion de brotes - Italia
    plot_outbreak_distribution(df_italy, 'Italia')
   

    ## Transfer learning con datos de España ##

    # Carga y preprocesamiento
    df_spain = load_and_preprocess('spain_data.csv')
    X_es = df_spain.drop('outbreak', axis=1)
    y_es = df_spain['outbreak']
   
    # Aplicar SMOTE para balancear las clases 
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_res, y_res = smote.fit_resample(X_es, y_es)
    
    # Escalar los datos después de SMOTE
    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)

    # Dividir X_res en entrenamiento y prueba
    X_train_es, X_test_es, y_train_es, y_test_es = train_test_split(X_res_scaled, y_res, test_size=0.2, random_state=42)

    # Reentrenar SOLO en el subconjunto de entrenamiento
    model_spain = transfer_learning(best_model, X_train_es, y_train_es)

    # Evaluar en datos NO vistos (X_test_es)
    y_pred_es = model_spain.predict(X_test_es)
    y_proba_es = model_spain.predict_proba(X_test_es)[:, 1]
    
    # Definir el umbral
    threshold = 0.5

    # Realizar la predicción ajustada
    y_pred_es_adjusted = (y_proba_es >= threshold).astype(int)
    
    # Metricas de España
    metrics_es = {
        'accuracy': accuracy_score(y_test_es, y_pred_es_adjusted),
        'precision': precision_score(y_test_es, y_pred_es_adjusted),
        'recall': recall_score(y_test_es, y_pred_es_adjusted),
        'f1': f1_score(y_test_es, y_pred_es_adjusted),
        'roc_auc': roc_auc_score(y_test_es, y_proba_es)
        }

    print("Métricas después de transfer learning:", metrics_es)

    # Guardar metricas en CSV    
    data_es = []  
    data_es.append(metrics_es)
    df_metrics_es = pd.DataFrame(data_es)
    df_metrics_es.to_csv('results/metrics_spain.csv', index=False)

    # Grafico de distribucion de brotes - España
    plot_outbreak_distribution(df_spain, 'España') 
        
    # Guardar modelo de España 
    joblib.dump(model_spain, 'results/model_spain.pkl')
    
    # Guardar scaler
    #scaler = model_spain.named_steps['scaler']  
    joblib.dump(scaler,'results/scaler_spain.pkl')

    print("\nProceso completado. Resultados guardados en la carpeta 'results'")