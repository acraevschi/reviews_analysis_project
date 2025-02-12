import json
from llamaapi import LlamaAPI

llama = LlamaAPI("LL-pANBVC0y8VoputpvhNkGbJEKDInrjte27wk3d1rbTHWCfrVlwgMxgKnUijSFTX6I")

df = ...

df[df["emotion"] == "anger"]


def fill_llama_request(emotion_type):
    api_request = {
        "model": "llama3.1-8b",
        "messages": [
            {
                "role": "user",
                "content": f"You are a text summarizer model of YouTube comments. Check the comments that are provided and extract any information that is useful to the creator of the video. Extract only the information that refers to the video creation aspect, rather than to the content of the video. Write it down in 1-2 paragraphs in the language that most of the comments are written in. Individual comments are separated by <sep>. You will be provided with only negative or only positive comments. I need you to write the response containing only the summary itself. Here are the '{emotion_type}'-only comments:",
            },
        ],
        "stream": False,
        "max_tokens": 1000,
        "temperature": 0.5,
    }
    return api_request


pos_summ_request = fill_llama_request("positive")
neg_summ_request = fill_llama_request("negative")

comments_lst = df[df["emotion"] == "joy"]["comment"].to_list()
comments_str = "<sep>".join(comments_lst)
pos_summ_request["messages"][0]["content"] += comments_str

comments_lst = df[df["emotion"].isin(["anger", "sadness", "fear"])]["comment"].tolist()
comments_str = "<sep>".join(comments_lst)
neg_summ_request["messages"][0]["content"] += comments_str


response = llama.run(pos_summ_request)
response = llama.run(neg_summ_request)

response.json()
