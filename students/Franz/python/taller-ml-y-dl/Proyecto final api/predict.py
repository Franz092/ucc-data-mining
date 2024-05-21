import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    "home_score": 7,
    "away_score": 0,
    "goal_difference": -7
}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, json=data, headers=headers)
print(response.json())
