import requests

def predict_match_result(home_team, away_team):
    url = 'http://127.0.0.1:5000/'  # Cambia la URL si es necesario
    data = {
        'home_team': home_team,
        'away_team': away_team
    }
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f'Error en la solicitud: {response.text}')
        return None

if __name__ == "__main__":
    home_team = 'England'
    away_team = 'Northern Ireland'
    result = predict_match_result(home_team, away_team)
    if result:
        print(f"Probabilidad de que {home_team} gane contra {away_team}: {result['probability']}")
