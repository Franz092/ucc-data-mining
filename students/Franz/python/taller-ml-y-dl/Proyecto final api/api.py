from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging

app = Flask(__name__)

# Configurar el registro de solicitudes
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "El servidor, despues de tanto está funcionando"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos de la solicitud
        data = request.get_json(force=True)
        
        # Validar y extraer las características
        if 'home_score' not in data or 'pena' not in data or 'away_score' not in data:
            return jsonify({'error': 'Datos de entrada incompletos'}), 400

        try:
            home_score = float(data['home_team'])
            away_score= float(data['away_team'])
            difference = float(data['difference'])
        except ValueError:
            return jsonify({'error': 'Los datos de entrada deben ser numericos'}), 400

        features = [home_score, away_score, difference]
        
        # Hacer la predicción
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1]

        return jsonify({'prediction': int(prediction), 'probability': probability})

    except Exception as e:
        app.logger.error(f"Error en la predicción: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
