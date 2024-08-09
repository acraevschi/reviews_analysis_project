import requests
import json

# URL of the Flask API endpoint
url = "http://127.0.0.1:5000/analyze"

# Data to be sent in the GET request
data = {
    "youtube_link": "https://www.youtube.com/watch?v=GY7HTeTWleY",
    "num_comments": 100,
}

# Send GET request
response = requests.get(url, json=data)

# Print the response from the server
if response.status_code == 200:
    print("Response JSON:\n", response.json())
else:
    print("Failed to get response. Status code:", response.status_code)
    print("Response content:", response.text)

response_json = response.json()
# Save response_json as a JSON file
with open("response.json", "w") as file:
    json.dump(response_json, file, indent=2)
