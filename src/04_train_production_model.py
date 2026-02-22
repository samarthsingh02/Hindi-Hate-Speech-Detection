import pandas as pd
import numpy as np
import os
import torch
import gc
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset

# --- Global Seed for Reproducibility ---
set_seed(42)

# --- Configuration ---
DATA_PATH = '../data/processed/unified_dataset.csv'
PRODUCTION_MODEL_DIR = '../models/final_production_model/'
LOG_FILE = os.path.join(PRODUCTION_MODEL_DIR, 'production_training_log.txt')

MODEL_NAME = 'google/muril-base-cased'
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 5
BEST_LR = 2e-5  # The winning LR from CV


def log_message(message):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': label
        }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    os.makedirs(PRODUCTION_MODEL_DIR, exist_ok=True)
    open(LOG_FILE, 'w').close()

    log_message("=" * 50)
    log_message("INITIATING FINAL PRODUCTION MODEL TRAINING")
    log_message("=" * 50)

    log_message("Loading dataset...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    df.dropna(subset=['text', 'label'], inplace=True)
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.strip() != '']

    texts = df['text'].values
    labels = df['label'].values

    # Create a 90/10 split for final training and early stopping validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=42
    )

    log_message(f"Training on {len(train_texts)} samples, Validating on {len(val_texts)} samples.")

    # Calculate class weights exclusively on the 90% training split
    class_weights_array = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights_array, dtype=torch.float)
    log_message(f"Production Class Weights: {class_weights_array}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = HateSpeechDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=os.path.join(PRODUCTION_MODEL_DIR, 'checkpoints'),
        num_train_epochs=EPOCHS,
        learning_rate=BEST_LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        fp16=True,
        report_to="none",
        save_total_limit=1
    )

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    log_message("\nCommencing Training...")
    trainer.train()

    log_message("\nEvaluating Final Model on 10% Holdout Data...")
    eval_results = trainer.evaluate()
    final_f1 = eval_results['eval_f1']
    final_acc = eval_results['eval_accuracy']

    log_message(f"Final Production F1-Score: {final_f1:.4f}")
    log_message(f"Final Production Accuracy: {final_acc:.4f}")

    # --- Exporting the Clean Production Model ---
    log_message("\nExporting model and tokenizer for deployment...")
    trainer.save_model(PRODUCTION_MODEL_DIR)
    tokenizer.save_pretrained(PRODUCTION_MODEL_DIR)

    log_message(f"SUCCESS: Production model saved to {PRODUCTION_MODEL_DIR}")
    clear_gpu_memory()


if __name__ == "__main__":
    main()