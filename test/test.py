import requests 

# https://your-heroku-app-name.herokuapp.com/predict
# http://localhost:5000/predict
resp = requests.post("http://localhost:5000/predict", files={'file': open('img/ship.jpg', 'rb')})

print(resp.text)
