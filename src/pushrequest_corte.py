import requests
import json

url = "https://forefire.univ-corse.fr/live/getRonan.php"

data = {
    "type": "Feature",
    "geometry": {
        "type": "Point",
        "coordinates": [9.0, 43.0]  # lon, lat
    },
    "properties": {
        "FRP": 35.6,
        "DATETIME": "2025-05-05T14:30:00Z",
        "satellite": "MODIS"
    }
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print("Status code:", response.status_code)
print("Response:", response.text)
