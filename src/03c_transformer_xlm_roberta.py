import pandas as pd
import numpy as np
import os
import torch
import gc
from torch import nn
from sklearn.model_selection import StratifiedKFold
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

# --- 1. Global Seed for Reproducibility ---
set_seed(42)

# --- Local Configuration & Paths ---
DATA_PATH = '../data/processed/unified_dataset.csv'
MODEL_DIR = '../models/transformers/xlm_roberta/'
LOG_FILE = os.path.join(MODEL_DIR, 'training_log.txt')

MODEL_NAME = 'xlm-roberta-base'
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATES = [2e-5, 3e-5, 5e-5]
N_SPLITS = 3


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

        # --- 2. Dynamic Padding Optimization ---
        # Removed hardcoded padding and tensor return.
        # The DataCollator will handle this dynamically per batch.
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
    os.makedirs(MODEL_DIR, exist_ok=True)
    open(LOG_FILE, 'w').close()

    log_message("Loading data...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8')

    df.dropna(subset=['text', 'label'], inplace=True)
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.strip() != '']

    texts = df['text'].values
    labels = df['label'].values

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    folds = list(skf.split(texts, labels))

    # Initialize Dynamic Padding Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ==========================================
    # PHASE 1: LR SCOUTING (FOLD 1 ONLY)
    # ==========================================
    log_message("\n" + "=" * 50)
    log_message("PHASE 1: LEARNING RATE SCOUTING (Fold 1 Only)")
    log_message("=" * 50)

    train_idx_f1, val_idx_f1 = folds[0]
    train_texts_f1, val_texts_f1 = texts[train_idx_f1], texts[val_idx_f1]
    train_labels_f1, val_labels_f1 = labels[train_idx_f1], labels[val_idx_f1]

    # --- 3. Data Leakage Fix ---
    # Class weights are now strictly calculated using the training data of the current fold.
    class_weights_array_f1 = compute_class_weight('balanced', classes=np.unique(train_labels_f1), y=train_labels_f1)
    class_weights_tensor_f1 = torch.tensor(class_weights_array_f1, dtype=torch.float)
    log_message(f"Fold 1 Training Class Weights: {class_weights_array_f1}")

    train_dataset_f1 = HateSpeechDataset(train_texts_f1, train_labels_f1, tokenizer, MAX_LEN)
    val_dataset_f1 = HateSpeechDataset(val_texts_f1, val_labels_f1, tokenizer, MAX_LEN)

    best_lr = None
    best_f1_fold1 = 0
    fold_1_results = {}

    for lr in LEARNING_RATES:
        log_message(f"\nScouting LR: {lr} on Fold 1...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

        training_args = TrainingArguments(
            output_dir=os.path.join(MODEL_DIR, f'scout_lr{lr}'),
            num_train_epochs=EPOCHS,
            learning_rate=lr,
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
            class_weights=class_weights_tensor_f1,
            model=model,
            args=training_args,
            train_dataset=train_dataset_f1,
            eval_dataset=val_dataset_f1,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train()
        eval_results = trainer.evaluate()
        eval_f1 = eval_results['eval_f1']

        fold_1_results[lr] = eval_f1
        log_message(f"Result for LR {lr}: F1 = {eval_f1:.4f}")

        if eval_f1 > best_f1_fold1:
            best_f1_fold1 = eval_f1
            best_lr = lr

        del model, trainer
        clear_gpu_memory()

    log_message(f"\n>>> SCOUTING COMPLETE. WINNING LR: {best_lr} (F1: {best_f1_fold1:.4f}) <<<")

    # ==========================================
    # PHASE 2: FULL CROSS-VALIDATION ON BEST LR
    # ==========================================
    log_message("\n" + "=" * 50)
    log_message(f"PHASE 2: EXECUTING REMAINING FOLDS WITH LR {best_lr}")
    log_message("=" * 50)

    all_fold_f1_scores = [best_f1_fold1]

    for fold_num in range(1, N_SPLITS):
        log_message(f"\n--- Starting Fold {fold_num + 1}/{N_SPLITS} ---")

        train_idx, val_idx = folds[fold_num]
        train_texts, val_texts = texts[train_idx], texts[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        # Recalculate pure weights for this specific training fold
        class_weights_array = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights_tensor = torch.tensor(class_weights_array, dtype=torch.float)
        log_message(f"Fold {fold_num + 1} Training Class Weights: {class_weights_array}")

        train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer, MAX_LEN)
        val_dataset = HateSpeechDataset(val_texts, val_labels, tokenizer, MAX_LEN)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

        training_args = TrainingArguments(
            output_dir=os.path.join(MODEL_DIR, f'final_fold{fold_num}'),
            num_train_epochs=EPOCHS,
            learning_rate=best_lr,
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

        trainer.train()
        eval_results = trainer.evaluate()
        fold_f1 = eval_results['eval_f1']
        all_fold_f1_scores.append(fold_f1)

        log_message(f"Fold {fold_num + 1} F1-Score: {fold_f1:.4f}")

        del model, trainer
        clear_gpu_memory()

    final_cv_f1 = np.mean(all_fold_f1_scores)

    log_message("\n" + "#" * 50)
    log_message("XLM-RoBERTa OPTIMIZED GRID SEARCH COMPLETE")
    log_message("#" * 50)
    log_message(f"Best Learning Rate Used: {best_lr}")
    log_message(f"Final 3-Fold CV F1-Score: {final_cv_f1:.4f}")

    results_df = pd.DataFrame([{
        'Model': 'XLM-RoBERTa (Optimized)',
        'Learning_Rate': best_lr,
        'CV_F1_Score': final_cv_f1,
        'Max_Len': MAX_LEN,
        'Class_Weights': 'Applied'
    }])

    csv_path = os.path.join(MODEL_DIR, 'xlm_roberta_optimized_cv_results.csv')
    results_df.to_csv(csv_path, index=False)
    log_message(f"Results successfully saved to: {csv_path}")


if __name__ == "__main__":
    main()