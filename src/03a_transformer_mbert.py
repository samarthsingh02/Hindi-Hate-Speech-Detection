import pandas as pd
import numpy as np
import os
import torch
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    EarlyStoppingCallback
from torch.utils.data import Dataset

# --- Local Configuration & Paths ---
DATA_PATH = '../data/processed/unified_dataset.csv'
MODEL_DIR = '../models/transformers/mbert/'

MODEL_NAME = 'bert-base-multilingual-cased'
MAX_LEN = 64
BATCH_SIZE = 16  # Reduced to fit local 6GB GPU VRAM
EPOCHS = 5
LEARNING_RATES = [2e-5, 3e-5, 5e-5]
N_SPLITS = 3


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
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


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


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8')

    # --- FIX 1: Clean NaNs and empty strings ---
    df.dropna(subset=['text', 'label'], inplace=True)
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.strip() != '']

    texts = df['text'].values
    labels = df['label'].values

    os.makedirs(MODEL_DIR, exist_ok=True)
    results_log = []

    print(f"Initializing Tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for lr in LEARNING_RATES:
        print(f"\n{'=' * 50}")
        print(f"STARTING GRID SEARCH: Learning Rate = {lr}")
        print(f"{'=' * 50}")

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

        # --- FIX 2: Track both F1 and Accuracy properly across folds ---
        fold_f1_scores = []
        fold_acc_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            print(f"\n--- Fold {fold + 1}/{N_SPLITS} for LR {lr} ---")

            train_texts, val_texts = texts[train_idx], texts[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]

            train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer, MAX_LEN)
            val_dataset = HateSpeechDataset(val_texts, val_labels, tokenizer, MAX_LEN)

            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

            training_args = TrainingArguments(
                output_dir=os.path.join(MODEL_DIR, f'checkpoints_lr{lr}_fold{fold}'),
                num_train_epochs=EPOCHS,
                learning_rate=lr,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE * 2,
                warmup_steps=500,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model='f1',
                fp16=True,
                report_to="none",
                save_total_limit=1
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
            )

            trainer.train()

            print(f"Evaluating Fold {fold + 1}...")
            eval_results = trainer.evaluate()

            fold_f1 = eval_results['eval_f1']
            fold_acc = eval_results['eval_accuracy']

            fold_f1_scores.append(fold_f1)
            fold_acc_scores.append(fold_acc)

            print(f"Fold {fold + 1} F1-Score: {fold_f1:.4f}")

            # --- FIX 3: Manually free up GPU VRAM before the next fold ---
            del model
            del trainer
            gc.collect()
            torch.cuda.empty_cache()

        avg_f1 = np.mean(fold_f1_scores)
        avg_acc = np.mean(fold_acc_scores)

        print(f"\n>>> Average CV F1-Score for LR {lr}: {avg_f1:.4f} <<<")

        results_log.append({
            'Model': 'mBERT',
            'Learning_Rate': lr,
            'CV_F1_Score': avg_f1,
            'CV_Accuracy': avg_acc
        })

    print("\n" + "#" * 50)
    print("mBERT GRID SEARCH COMPLETE")
    print("#" * 50)
    results_df = pd.DataFrame(results_log).sort_values(by='CV_F1_Score', ascending=False)
    print(results_df.to_string(index=False))

    csv_path = os.path.join(MODEL_DIR, 'mbert_cv_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results successfully saved to: {csv_path}")


if __name__ == "__main__":
    main()