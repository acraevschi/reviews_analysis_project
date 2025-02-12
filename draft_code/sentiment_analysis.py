from draft_models.youtube_comments_api import get_comments
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# LINK = "https://www.youtube.com/watch?v=IIGCq3PKsgw"  # 893k views
# LINK = "https://www.youtube.com/watch?v=Xwq8Vp8j7Lg"  # 223k views
# LINK = "https://www.youtube.com/watch?v=NUAovVNl8zw"  # 299k views

# LINK = "https://www.youtube.com/watch?v=UpmwhkNg5Dw"  # other channel

LINK = "https://www.youtube.com/watch?v=xjquKKUEExo"  # VisualPolitik

# LINK = "https://www.youtube.com/watch?v=vj71yGp-8WM"  # WIRED

# LINK = "https://www.youtube.com/watch?v=36F6mBpZZ4E"  # Дробышевский

# LINK = "https://www.youtube.com/watch?v=-_wMvLpOnPQ"  # ФБК


tokenizer = AutoTokenizer.from_pretrained(
    "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
)

df = get_comments(LINK, num_comments=200)
df.reset_index(inplace=True, drop=True)

tokenized_comments = tokenizer(
    df["comment"].tolist(),
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

batch_size = 16
# Batchify the input
data = TensorDataset(
    tokenized_comments["input_ids"], tokenized_comments["attention_mask"]
)
dataloader = DataLoader(data, batch_size=batch_size)

logits = []
preds = []
for batch in tqdm(dataloader):
    input_ids, attention_mask = batch
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits.extend(outputs.logits)
        preds.extend(outputs.logits.argmax(dim=1).tolist())
    break

preds
int_to_label = {"0": "negative", "1": "neutral", "2": "positive"}
preds_labels = [int_to_label[str(pred)] for pred in preds]

# Create histogram of preds_labels
plt.hist(preds_labels, bins=4, rwidth=0.8)
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.title("Distribution of Sentiments")
plt.show()


for comment, pred in zip(df["comment"], preds_labels):
    print(comment)
    print(pred)
    print("___" * 20)


df["emotion"] = preds_labels
