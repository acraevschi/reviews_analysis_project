import requests
import json

request = {
    "messages": [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "How do I init and update a git submodule?"},
    ],
    "temperature": 0.7,
    "max_tokens": -1,
    "stream": True,
}


address = "http://localhost:1234/v1/chat/completions"

response = requests.post(address, json=request)
# how to convert the response to a JSON object
response_json = json.loads(response.text.strip())
