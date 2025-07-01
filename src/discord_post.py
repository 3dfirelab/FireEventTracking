import requests

url = 'https://api.sedoo.fr/aeris-euburn-silex-rest/discord/sendMessage'
headers = {
'accept': '*/*',
'Content-Type': 'application/json',
}
data = '"Test RONAN!!"' 

response = requests.post(url, headers=headers, data=data)

print(response.status_code)
print(response.text)
