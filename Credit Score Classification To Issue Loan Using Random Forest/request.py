import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'a':19114.12, 'b':1824.843333, 'c': 2, 'd':2, 'e':9, 'f':2, 'g':12, 'h':3, 'i':3, 'j':250, 'k':200, 'l':310})

print(r.json())