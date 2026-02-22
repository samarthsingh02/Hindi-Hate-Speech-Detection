import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- Configuration ---
DATA_PATH = '../data/processed/unified_dataset.csv'
MURIL_DIR = '../models/transformers/muril/scout_lr2e-05'
XLM_DIR = '../models/transformers/xlm_roberta/scout_lr2e-05'

MAX_LEN = 128
BATCH_SIZE = 16
N_SPLITS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


def get_latest_checkpoint(base_dir):
    """Finds the most recent Hugging Face checkpoint folder inside the directory."""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith('checkpoint')]
    if not subdirs:
        raise FileNotFoundError(f"No checkpoint folder found in {base_dir}")
    return max(subdirs, key=os.path.getmtime)


def get_model_probabilities(model_path, model_name, texts):
    """Loads a model and generates raw probabilities for Class 1 (Hate Speech)."""
    print(f"\nLoading {model_name} from: {model_path}")

    # Check if the base model name is needed for the tokenizer
    hf_model_name = 'google/muril-base-cased' if 'muril' in model_name.lower() else 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.to(DEVICE)
    model.eval()

    dataset = InferenceDataset(texts, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_probs = []

    print(f"Generating probabilities for {model_name}...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Apply softmax to convert logits to percentages (0.0 to 1.0)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

    del model
    torch.cuda.empty_cache()

    return np.array(all_probs)


def main():
    print("Loading validation data...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    df.dropna(subset=['text', 'label'], inplace=True)
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.strip() != '']

    texts = df['text'].values
    labels = df['label'].values

    # Recreate the exact Fold 1 split
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    folds = list(skf.split(texts, labels))
    _, val_idx = folds[0]

    val_texts = texts[val_idx]
    val_labels = labels[val_idx]

    print(f"Validation Set Size: {len(val_texts)} texts")

    # Locate the checkpoint folders
    muril_ckpt = get_latest_checkpoint(MURIL_DIR)
    xlm_ckpt = get_latest_checkpoint(XLM_DIR)

    # Extract probabilities
    muril_probs = get_model_probabilities(muril_ckpt, "MuRIL", val_texts)
    xlm_probs = get_model_probabilities(xlm_ckpt, "XLM-RoBERTa", val_texts)

    # --- 1. Soft Voting Ensemble ---
    ensemble_probs = (muril_probs + xlm_probs) / 2.0

    # --- 2. Threshold Optimization ---
    print("\n" + "=" * 50)
    print("THRESHOLD OPTIMIZATION RESULTS (ENSEMBLE)")
    print("=" * 50)

    best_f1 = 0.0
    best_thresh = 0.5
    best_metrics = {}

    for thresh in np.arange(0.30, 0.71, 0.01):
        # If probability is greater than threshold, classify as 1 (Hate)
        preds = (ensemble_probs >= thresh).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, preds, average='binary', zero_division=0
        )
        acc = accuracy_score(val_labels, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {'precision': precision, 'recall': recall, 'acc': acc}

    print(f"Optimal Threshold : {best_thresh:.2f}")
    print(f"Ensemble F1-Score : {best_f1:.4f}")
    print(f"Ensemble Recall   : {best_metrics['recall']:.4f}")
    print(f"Ensemble Precision: {best_metrics['precision']:.4f}")
    print(f"Ensemble Accuracy : {best_metrics['acc']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()