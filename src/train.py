import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import yaml
import os
import sys

def load_config(config_path="../config.yaml"):
    """Carga el archivo config.yaml con manejo de errores."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró {config_path}. Asegúrate de que existe en el directorio raíz.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error al parsear config.yaml: {e}")
        sys.exit(1)

def main():
    # Cargar configuración
    config = load_config()

    # Verificar configuración
    if not config.get('data') or not config.get('preprocessing') or not config.get('model'):
        print("Error: config.yaml incompleto. Revisa las claves 'data', 'preprocessing', 'model'.")
        sys.exit(1)

    # Paso 1: Cargar datos
    try:
        data_url = config['data']['url']
        columns = config['data']['columns']
        df = pd.read_csv(data_url, header=None, names=columns)
        print(f"Datos cargados: {df.shape}")
    except Exception as e:
        print(f"Error al cargar el dataset desde {data_url}: {e}")
        sys.exit(1)

    # Paso 2: Preprocesamiento
    # Manejo de nulos
    if df.isnull().sum().sum() > 0:
        print("Advertencia: Datos con valores nulos. Aplicando dropna.")
        df = df.dropna()
    
    # Separar features y target
    try:
        X = df.drop('class', axis=1)
        y = df['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
        if y.isnull().any():
            print("Error: Falló la codificación de la columna 'class'. Revisa los datos.")
            sys.exit(1)
    except KeyError as e:
        print(f"Error: Columna 'class' no encontrada en el dataset: {e}")
        sys.exit(1)

    # Escalamiento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # División train/test
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=config['preprocessing']['test_size'], 
            random_state=config['preprocessing']['random_state']
        )
    except KeyError as e:
        print(f"Error en parámetros de preprocesamiento: {e}")
        sys.exit(1)

    # Paso 3: Entrenamiento
    try:
        model_params = config['model']['params']
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error al entrenar el modelo: {e}")
        sys.exit(1)

    # Paso 4: Evaluación
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}")

    # Paso 5: Tracking con MLflow
    try:
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        with mlflow.start_run(run_name="Iris_Classification"):
            # Registrar parámetros
            mlflow.log_params(model_params)
            mlflow.log_param("test_size", config['preprocessing']['test_size'])
            
            # Registrar métricas
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_macro", f1_macro)
            
            # Inferir firma
            signature = infer_signature(X_train, model.predict(X_train))
            
            # Registrar modelo
            mlflow.sklearn.log_model(model, "model", signature=signature)
            print(f"Modelo registrado en MLflow en {config['mlflow']['tracking_uri']}")
    except Exception as e:
        print(f"Error con MLflow: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()