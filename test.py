from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Cargar el modelo entrenado
modelo = joblib.load('c:\\Users\\chami\\Downloads\\modelo_lightgbm_refinado.pkl')

# Inicializar Flask
app = Flask(__name__)

# Agregar esta función antes de la ruta /predict
@app.route('/')
def home():
    return "Bienvenido a la API de predicción"

# Definir la ruta para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del POST request en formato JSON
    data = request.get_json(force=True)
    
    # Convertir los datos a un DataFrame de Pandas
    data_df = pd.DataFrame([data])
    
    # Hacer la predicción
    prediccion = modelo.predict(data_df)
    
    # Retornar la predicción como respuesta JSON
    return jsonify({'prediction': prediccion.tolist()})

# Ejecutar el servidor Flask
if __name__ == '__main__':
    app.run(debug=True, port=5001)



import requests

# Definir los datos de entrada
data = {
    "ShotsOnTarget_H": 4,
    "PPG_Home": 1.5,
    "Corners_H_FT": 6,
    "PPG_Away": 1.2,
    "ShotsOffTarget_H": 3,
    "Corners_A_FT": 5,
    "Shots_A": 8,
    "PPG_Home_Pre": 1.4,
    "XG_Home_Pre": 1.3,
    "PPG_Away_Pre": 1.1
}

# Hacer una solicitud POST a la API
response = requests.post("http://127.0.0.1:5001/predict", json=data)

# Mostrar la predicción
print(response.json())
