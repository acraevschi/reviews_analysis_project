import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import f1_score, roc_auc_score
import json
from transformers import EarlyStoppingCallback

# 1. Load Data
# Assuming `raw_data` is the JSON list you provided
with open("classification_data/labeled_dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)
dataset = Dataset.from_list(raw_data)

LABELS = ["is_request", "is_question", "is_highlight", "is_feedback", "is_spam"]
MODEL_ID = "jhu-clsp/mmBERT-base"

# 2. Initialization
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(LABELS),
    problem_type="multi_label_classification",
    attn_implementation="flash_attention_2",  # Accelerates training and reduces VRAM
    id2label={i: label for i, label in enumerate(LABELS)},
    label2id={label: i for i, label in enumerate(LABELS)}
)

try:
    model.to("cuda")
except:
    pass

# 3. Preprocessing
def preprocess_function(examples):
    # Combine title and comment to give the model maximum context
    description = examples["video_description"]
    decription_trimmed = description if len(description) < 500 else description[:500] + "..."
    texts = [f"Title: {t}\nDescription: {d}\nComment: {c}" for t, d, c in zip(examples["video_title"], decription_trimmed, examples["comment"])]
    
    # Tokenize texts
    tokenized = tokenizer(
        texts, 
        padding="max_length", 
        truncation=True, 
        max_length=2048
    )
    
    # Multi-label classification requires labels to be formatted as a float array
    labels_matrix = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
    for i, label in enumerate(LABELS):
        labels_matrix[:, i] = examples[label]
        
    tokenized["labels"] = labels_matrix.tolist()
    return tokenized

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# Split into train and evaluation sets
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=97)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# 4. Evaluation Metrics
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    
    # Apply sigmoid to convert raw logits to probabilities
    probs = 1 / (1 + np.exp(-logits))
    
    # Using a flat 0.5 threshold for training evaluation. 
    # (In production, you should calibrate this threshold per class)
    predictions = (probs > 0.5).astype(int)
    
    macro_f1 = f1_score(labels, predictions, average="macro")
    roc_auc = roc_auc_score(labels, probs, average="macro")
    
    return {
        "macro_f1": macro_f1, 
        "roc_auc": roc_auc
    }

# 5. Training Setup
training_args = TrainingArguments(
    output_dir="./modernbert-youtube-comments",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    logging_steps=50,
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_accumulation_steps=2,
    num_train_epochs=4,
    weight_decay=0.01,
    bf16=True,
    load_best_model_at_end=True,
    save_total_limit=5,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# 6. Run Fine-Tuning
if __name__ == "__main__":
    trainer.train()
    
    # Save the final model and tokenizer
    trainer.save_model("./modernbert-youtube-comments-final")
    tokenizer.save_pretrained("./modernbert-youtube-comments-final")
    print("Training complete! Model saved.")