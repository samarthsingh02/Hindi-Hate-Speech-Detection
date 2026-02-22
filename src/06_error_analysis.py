import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- Configuration ---
DATA_PATH = '../data/processed/unified_dataset.csv'
MURIL_DIR = '../models/transformers/muril/scout_lr2e-05'
XLM_DIR = '../models/transformers/xlm_roberta/scout_lr2e-05'
OUTPUT_DIR = '../reports/'

MAX_LEN = 128
BATCH_SIZE = 16
N_SPLITS = 3
OPTIMAL_THRESHOLD = 0.50
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    muril_ckpt = get_latest_checkpoint(MURIL_DIR)
    xlm_ckpt = get_latest_checkpoint(XLM_DIR)

    muril_probs = get_model_probabilities(muril_ckpt, "MuRIL", val_texts)
    xlm_probs = get_model_probabilities(xlm_ckpt, "XLM-RoBERTa", val_texts)

    ensemble_probs = (muril_probs + xlm_probs) / 2.0
    preds = (ensemble_probs >= OPTIMAL_THRESHOLD).astype(int)

    # Build the analysis DataFrame
    analysis_df = pd.DataFrame({
        'Text': val_texts,
        'True_Label': val_labels,
        'Predicted_Label': preds,
        'MuRIL_Prob': np.round(muril_probs, 4),
        'XLM_Prob': np.round(xlm_probs, 4),
        'Ensemble_Prob': np.round(ensemble_probs, 4)
    })

    # --- 2. Add Disagreement Flag ---
    analysis_df['Model_Disagreement'] = (analysis_df['MuRIL_Prob'] >= OPTIMAL_THRESHOLD) != (
                analysis_df['XLM_Prob'] >= OPTIMAL_THRESHOLD)

    # Isolate the errors
    errors_df = analysis_df[analysis_df['True_Label'] != analysis_df['Predicted_Label']].copy()

    # Classify the error types
    errors_df['Error_Type'] = ''
    errors_df.loc[(errors_df['True_Label'] == 0) & (
                errors_df['Predicted_Label'] == 1), 'Error_Type'] = 'False Positive (Hallucination)'
    errors_df.loc[(errors_df['True_Label'] == 1) & (
                errors_df['Predicted_Label'] == 0), 'Error_Type'] = 'False Negative (Missed Hate)'

    # --- 1. Confidence Margin Logic ---
    errors_df['Raw_Confidence'] = np.where(
        errors_df['Predicted_Label'] == 1,
        errors_df['Ensemble_Prob'],
        1 - errors_df['Ensemble_Prob']
    )

    # Sort by how confident the model was in being wrong
    errors_df = errors_df.sort_values(by='Raw_Confidence', ascending=False)

    output_file = os.path.join(OUTPUT_DIR, 'ensemble_error_analysis.csv')
    errors_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    # --- 3. Count Severe Misses ---
    severe_fn = len(
        errors_df[(errors_df['Error_Type'] == 'False Negative (Missed Hate)') & (errors_df['Ensemble_Prob'] < 0.2)])
    disagreement_errors = len(errors_df[errors_df['Model_Disagreement'] == True])

    print("\n" + "=" * 50)
    print(f"Total Validation Texts : {len(val_texts)}")
    print(f"Total Errors Made      : {len(errors_df)}")
    print(f"False Positives        : {len(errors_df[errors_df['Error_Type'].str.contains('False Positive')])}")
    print(f"False Negatives        : {len(errors_df[errors_df['Error_Type'].str.contains('False Negative')])}")
    print("-" * 50)
    print(f"Errors with Model Disagreement : {disagreement_errors}")
    print(f"Severe False Negatives (<0.2)  : {severe_fn}")
    print(f"Report successfully saved to   : {output_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()