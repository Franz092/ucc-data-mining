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

# Función para predecir la probabilidad de que el equipo local gane contra el equipo visitante
def predict_probability(home_team, away_team):
    # Generar características relevantes del partido (si es necesario)
    match_features = generate_match_features(home_team, away_team)
    
    # Realizar la predicción de probabilidad
    probability = model.predict_proba([match_features])[0][1]
    return probability

# Función para generar características relevantes del partido
def generate_match_features(home_team, away_team):
    # Ejemplo de generación de características relevantes del partido
    # Aquí puedes agregar más características según sea necesario
    return [get_team_rank(home_team), get_team_rank(away_team)]

# Ejemplo de función para obtener el rango del equipo
def get_team_rank(team_name):
    # Lógica para obtener el rango del equipo desde una base de datos o API externa
    # Aquí simplemente se devuelve un valor aleatorio para fines de demostración
    return np.random.randint(1, 20)

@app.route('/')
def home():
    return "El servidor, después de tanto está funcionando 2"

@app.route('/predict_probability', methods=['POST'])
def predict_probability_route():
    try:
        # Obtener los datos de la solicitud
        data = request.get_json(force=True)
        
        # Validar y extraer los nombres de los equipos
        if 'home_team' not in data or 'away_team' not in data:
            return jsonify({'error': 'Nombres de equipos incompletos'}), 400

        home_team = data['home_team']
        away_team = data['away_team']
        
        # Realizar la predicción de probabilidad
        probability = predict_probability(home_team, away_team)

        return jsonify({'home_team': home_team, 'away_team': away_team, 'probability': probability})

    except Exception as e:
        app.logger.error(f"Error en la predicción de probabilidad: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
