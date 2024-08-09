from youtube_comments_api import get_comments
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd
import math
from torch.utils.data import DataLoader, TensorDataset


LINK = "https://www.youtube.com/watch?v=IIGCq3PKsgw"  # 893k views
# LINK = "https://www.youtube.com/watch?v=Xwq8Vp8j7Lg"  # 223k views
# LINK = "https://www.youtube.com/watch?v=NUAovVNl8zw"  # 299k views

# LINK = "https://www.youtube.com/watch?v=UpmwhkNg5Dw"  # other channel

# LINK = "https://www.youtube.com/watch?v=xjquKKUEExo"  # VisualPolitik

# LINK = "https://www.youtube.com/watch?v=2CEgqX15a7s"  # WIRED

# LINK = "https://www.youtube.com/watch?v=36F6mBpZZ4E"  # Дробышевский

# LINK = "https://www.youtube.com/watch?v=-_wMvLpOnPQ"  # ФБК

model_name = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"

df = get_comments(LINK, num_comments=200)
df = df[df["refers_to"].isna()]  # Filter out comments that refer to other comments
df.reset_index(inplace=True, drop=True)
classifier = pipeline("zero-shot-classification", model=model_name)

batch_size = 8
num_batches = math.ceil(len(df) / batch_size)

candidate_labels = ["positive", "negative", "sarcastic", "neutral", "question", "spam"]
hypothesis_template = "This YouTube comment is {}"


def reformat_results(output):
    sequence = output["sequence"]
    labels = output["labels"]
    scores = output["scores"]
    new_dict = {"comment": sequence}
    new_dict.update({labels[i]: scores[i] for i in range(len(labels))})
    # Create the new dictionary
    return new_dict


model_results = pd.DataFrame(columns=["comment"] + candidate_labels)

for i in tqdm(range(num_batches)):
    batch_comments = df["comment"].tolist()[i * batch_size : (i + 1) * batch_size]
    output = classifier(
        batch_comments,
        candidate_labels,
        hypothesis_template=hypothesis_template,
        multi_label=False,
    )
    reformat_output = [reformat_results(result) for result in output]
    df_results = pd.DataFrame(reformat_output)
    model_results = pd.concat([model_results, df_results], axis=0)
    if i == 3:
        break

model_results.reset_index(inplace=True, drop=True)
model_results
for i in range(len(model_results)):
    print(model_results.iloc[i]["comment"])
    print(model_results.iloc[i][candidate_labels])
    print("___" * 20)
