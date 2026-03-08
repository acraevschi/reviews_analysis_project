import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, roc_auc_score
from scipy.special import expit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import json

# 1. Load Data
with open("classification_data/labeled_dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

LABELS = ["is_request", "is_question", "is_highlight", "is_feedback", "is_spam"]
MODEL_ID = "FacebookAI/xlm-roberta-base"

# Extract labels for stratification
all_labels = np.array([[row[label] for label in LABELS] for row in raw_data])

# 2. Multi-label Iterative Stratified Split
# This ensures train and eval sets have identical class proportions
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=97)
train_indices, eval_indices = next(msss.split(np.zeros(len(all_labels)), all_labels))

train_raw = [raw_data[i] for i in train_indices]
eval_raw = [raw_data[i] for i in eval_indices]

train_dataset = Dataset.from_list(train_raw)
eval_dataset = Dataset.from_list(eval_raw)

# Calculate dynamic class weights based ONLY on the training set
train_labels = np.array([[row[label] for label in LABELS] for row in train_raw])
pos_counts = train_labels.sum(axis=0)
neg_counts = len(train_labels) - pos_counts

pos_weights_array = neg_counts / (pos_counts + 1e-5) 
pos_weights_array = pos_weights_array / sum(pos_weights_array) * 5 + 1

pos_weights_tensor = torch.tensor(pos_weights_array, dtype=torch.float32)

print(f"Calculated Positive Class Weights: {pos_weights_tensor.tolist()}")

# 3. Initialization
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(LABELS),
    problem_type="multi_label_classification",
    # attn_implementation="flash_attention_2",
    dtype=torch.bfloat16,
    id2label={i: label for i, label in enumerate(LABELS)},
    label2id={label: i for i, label in enumerate(LABELS)}
)

# 4. Preprocessing
def preprocess_function(examples):
    description = examples["video_description"]
    # Handle both single strings and lists of strings (batched mapping)
    if isinstance(description, list):
        desc_trimmed = [d if len(d) < 500 else d[:500] + "..." for d in description]
    else:
        desc_trimmed = description if len(description) < 500 else description[:500] + "..."
        desc_trimmed = [desc_trimmed]
        
    texts = [
        f"Title: {t}\nDescription: {d}\nComment: {c}" 
        for t, d, c in zip(examples["video_title"], desc_trimmed, examples["comment"])
    ]
    
    tokenized = tokenizer(
        texts, 
        # padding="max_length", 
        truncation=True, 
        max_length=512
    )
    
    labels_matrix = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
    for i, label in enumerate(LABELS):
        labels_matrix[:, i] = examples[label]
    
    if np.isnan(labels_matrix).any():
        raise ValueError("NaN detected in labels! Check your JSON for missing annotations (null values).")
        
    tokenized["labels"] = labels_matrix.tolist()
    return tokenized

train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

# 5. Evaluation Metrics
def compute_metrics(eval_preds):
    logits, labels = eval_preds

    probs = expit(logits)
    predictions = (probs > 0.5).astype(int)
    
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    
    try:
        roc_auc = roc_auc_score(labels, probs, average="macro")
    except ValueError:
        roc_auc = float("nan")
    
    return {
        "macro_f1": macro_f1, 
        "roc_auc": roc_auc
    }

# 6. Custom Trainer with Weighted Loss
# Update your custom trainer class
class WeightedTrainer(Trainer):
    def __init__(self, *args, pos_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weights = pos_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        logits = outputs.logits
        
        if self.pos_weights is not None:
            # Ensure weights are on the correct device
            weights = self.pos_weights.to(logits.device)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=weights)
        else:
            loss_fct = nn.BCEWithLogitsLoss()
            
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# 7. Training Setup
training_args = TrainingArguments(
    output_dir="./xlm_roberta-youtube-comments",
    eval_strategy="steps",
    eval_delay=1000,
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=100,
    # optim="adamw_torch",
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_accumulation_steps=2,
    num_train_epochs=10,
    weight_decay=0.03,
    # max_grad_norm=1.0,
    bf16=True, # causes problems for some reason
    # fp16=True,
    load_best_model_at_end=True,
    save_total_limit=5,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    report_to="none",
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    pos_weights=pos_weights_tensor
)

# 8. Run Fine-Tuning
if __name__ == "__main__":
    trainer.train()
    
    # Save the final model and tokenizer
    trainer.save_model("./xlm_roberta-youtube-comments")
    # tokenizer.save_pretrained("./modernbert-youtube-comments-final")
    print("Training complete! Model saved.")
