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
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith('checkpoint')]
    if not subdirs:
        raise FileNotFoundError(f"No checkpoint folder found in {base_dir}")
    return max(subdirs, key=os.path.getmtime)


def get_model_probabilities(model_path, model_name, texts):
    print(f"Loading {model_name} from: {model_path}")
    hf_model_name = 'google/muril-base-cased' if 'muril' in model_name.lower() else 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.to(DEVICE)
    model.eval()

    dataset = InferenceDataset(texts, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Generating {model_name} Probs"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
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

    # Load models and get predictions (~5 mins total)
    muril_ckpt = get_latest_checkpoint(MURIL_DIR)
    xlm_ckpt = get_latest_checkpoint(XLM_DIR)

    muril_probs = get_model_probabilities(muril_ckpt, "MuRIL", val_texts)
    xlm_probs = get_model_probabilities(xlm_ckpt, "XLM-RoBERTa", val_texts)

    print("\n" + "=" * 50)
    print("INITIATING CPU GRID SEARCH (WEIGHTS & THRESHOLDS)")
    print("=" * 50)

    best_f1 = 0.0
    best_muril_weight = 0.5
    best_thresh = 0.5
    best_metrics = {}

    # Test MuRIL weights from 0.0 to 1.0 (in increments of 0.05)
    for w_muril in np.arange(0.0, 1.05, 0.05):
        w_xlm = 1.0 - w_muril

        # Calculate the new weighted probability
        weighted_probs = (muril_probs * w_muril) + (xlm_probs * w_xlm)

        # Test every threshold for this specific weight
        for thresh in np.arange(0.30, 0.71, 0.01):
            preds = (weighted_probs >= thresh).astype(int)

            precision, recall, f1, _ = precision_recall_fscore_support(
                val_labels, preds, average='binary', zero_division=0
            )

            if f1 > best_f1:
                best_f1 = f1
                best_muril_weight = w_muril
                best_thresh = thresh
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'acc': accuracy_score(val_labels, preds)
                }

    print(f"OPTIMAL COMBINATION FOUND:")
    print(f"MuRIL Weight      : {best_muril_weight * 100:.1f}%")
    print(f"XLM-RoBERTa Weight: {(1.0 - best_muril_weight) * 100:.1f}%")
    print(f"Optimal Threshold : {best_thresh:.2f}")
    print("-" * 50)
    print(f"New Peak F1-Score : {best_f1:.4f}")
    print(f"Recall            : {best_metrics['recall']:.4f}")
    print(f"Precision         : {best_metrics['precision']:.4f}")
    print(f"Accuracy          : {best_metrics['acc']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()