from youtube_comments_api import get_comments
from tqdm import tqdm
from detoxify import Detoxify
import pandas as pd
import math


def analyze_toxicity(LINK, NUM_COMMENTS=100):
    try:
        model = Detoxify("multilingual", device="cuda")
    except:
        model = Detoxify("multilingual", device="cpu")

    df = get_comments(LINK, num_comments=NUM_COMMENTS)
    df.reset_index(inplace=True, drop=True)

    batch_size = 16
    num_batches = math.ceil(len(df) / batch_size)
    toxicity_results = pd.DataFrame()
    for i in tqdm(range(num_batches), desc="Classifying toxicity"):
        batch_comments = df["comment"].tolist()[i * batch_size : (i + 1) * batch_size]
        batch_toxicity_results = model.predict(batch_comments)
        batch_toxicity_results = pd.DataFrame(batch_toxicity_results).round(2)
        toxicity_results = pd.concat([toxicity_results, batch_toxicity_results])

    toxicity_results.reset_index(inplace=True, drop=True)

    df = pd.concat([df, toxicity_results], axis=1)
    toxic_comments = df[df["toxicity"] > 0.5]
    prop_toxic = len(toxic_comments) / len(df)
    # first element is the proportion of toxic comments out of all the comments
    toxic_comments_dict = {
        "prop_toxic": prop_toxic,
        "num_comments": len(toxic_comments),
        "toxic_comments": toxic_comments.sort_values(by="toxicity", ascending=False)[
            "comment"
        ].tolist()[:5],
        "breakdown": {},
    }
    # Add prop of comments by type of toxicity (severe_toxicity, obscene, identity_attack, insult, threat, sexual_explicit)
    for col in [
        "severe_toxicity",
        "obscene",
        "identity_attack",
        "insult",
        "threat",
        "sexual_explicit",
    ]:
        toxic_comments_col = toxic_comments[toxic_comments[col] > 0.5]
        comments_lst = toxic_comments_col.sort_values(by=col, ascending=False)[
            "comment"
        ].tolist()[:5]
        prop_toxic_col = len(toxic_comments_col) / len(toxic_comments)
        toxic_comments_dict["breakdown"][col] = {
            "prop": prop_toxic_col,
            "num_comments": len(toxic_comments_col),
            "comments": comments_lst,
        }
    return toxic_comments_dict
