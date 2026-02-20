import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from collections import Counter

# --- Configuration & Paths ---
DATA_PATH = '../data/processed/unified_dataset.csv'
FASTTEXT_PATH = '../data/embeddings/cc.hi.300.vec'
MODEL_DIR = '../../models/dl_variants/'

# Hyperparameters for the Grid
MAX_LENS = [50, 100]
DROPOUTS = [0.3, 0.5]
EMBEDDING_DIM = 300
BATCH_SIZE = 128
EPOCHS = 3  # Kept low for the 2-day deadline!
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- 1. Data Prep & Vocabulary ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]).split()
        # Convert words to indices, using 1 for unknown words, 0 for padding
        encoded = [self.vocab.get(w, 1) for w in text]
        # Pad or truncate
        if len(encoded) < self.max_len:
            encoded = encoded + [0] * (self.max_len - len(encoded))
        else:
            encoded = encoded[:self.max_len]

        return torch.tensor(encoded, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)


def build_vocab(texts, max_words=30000):
    counter = Counter()
    for text in texts:
        counter.update(str(text).split())
    # 0 is padding, 1 is unknown
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(max_words - 2):
        vocab[word] = len(vocab)
    return vocab


def load_fasttext_matrix(vocab, filepath):
    print("Loading FastText embeddings (this takes a minute)...")
    embedding_matrix = np.random.normal(scale=0.6, size=(len(vocab), EMBEDDING_DIM))
    embedding_matrix[0] = np.zeros(EMBEDDING_DIM)  # Padding is zero

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.rstrip().split(' ')
                word = values[0]
                if word in vocab:
                    vector = np.asarray(values[1:], dtype='float32')
                    embedding_matrix[vocab[word]] = vector
    except FileNotFoundError:
        print(f"WARNING: FastText file not found at {filepath}. Falling back to Random Embeddings.")

    return torch.tensor(embedding_matrix, dtype=torch.float32)


# --- 2. Model Architectures ---

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_dim)
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attn_weights * lstm_output, dim=1)
        return context_vector


class BaseHateModel(nn.Module):
    def __init__(self, model_type, vocab_size, embed_matrix, dropout_rate):
        super(BaseHateModel, self).__init__()
        self.model_type = model_type

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        if embed_matrix is not None:
            self.embedding.weight = nn.Parameter(embed_matrix)
            self.embedding.weight.requires_grad = False  # Freeze pretrained weights

        self.dropout = nn.Dropout(dropout_rate)

        # Architecture routing
        if model_type == 'CNN' or model_type == 'CNN+LSTM':
            self.conv1 = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=128, kernel_size=3)
            self.conv2 = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=128, kernel_size=5)

        if 'LSTM' in model_type:
            lstm_in = 256 if model_type == 'CNN+LSTM' else EMBEDDING_DIM
            self.lstm = nn.LSTM(lstm_in, 128, bidirectional=True, batch_first=True)

        if 'Attention' in model_type:
            self.attention = Attention(256)  # 128 * 2 for bidirectional

        # Fully Connected Classifier
        fc_in = {
            'CNN': 256,
            'BiLSTM': 256,
            'BiLSTM+Attention': 256,
            'CNN+LSTM': 256
        }[model_type]

        self.fc = nn.Linear(fc_in, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = self.dropout(x)

        if self.model_type == 'CNN':
            x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len) for Conv1d
            c1 = torch.relu(self.conv1(x))
            c2 = torch.relu(self.conv2(x))
            c1 = torch.max(c1, dim=2)[0]
            c2 = torch.max(c2, dim=2)[0]
            out = torch.cat((c1, c2), dim=1)

        elif self.model_type == 'BiLSTM':
            lstm_out, _ = self.lstm(x)
            out = lstm_out[:, -1, :]  # Take last hidden state

        elif self.model_type == 'BiLSTM+Attention':
            lstm_out, _ = self.lstm(x)
            out = self.attention(lstm_out)

        elif self.model_type == 'CNN+LSTM':
            x = x.permute(0, 2, 1)
            c1 = torch.relu(self.conv1(x))
            c2 = torch.relu(self.conv2(x))
            # Truncate to match sequences and concatenate
            min_len = min(c1.size(2), c2.size(2))
            c_out = torch.cat((c1[:, :, :min_len], c2[:, :, :min_len]), dim=1)
            c_out = c_out.permute(0, 2, 1)  # Back to (batch, seq, channels)
            lstm_out, _ = self.lstm(c_out)
            out = lstm_out[:, -1, :]

        out = self.dropout(out)
        out = self.sigmoid(self.fc(out))
        return out.squeeze()


# --- 3. Training & Evaluation Engine ---

def train_and_evaluate():
    print(f"Using device: {DEVICE}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    texts, labels = df['text'].values, df['label'].values
    vocab = build_vocab(texts)
    vocab_size = len(vocab)

    fasttext_matrix = load_fasttext_matrix(vocab, FASTTEXT_PATH)

    models = ['CNN', 'BiLSTM', 'BiLSTM+Attention', 'CNN+LSTM']
    embeddings = [('Random', None), ('FastText', fasttext_matrix)]

    results = []

    for model_name in models:
        for embed_name, embed_matrix in embeddings:
            for max_len in MAX_LENS:
                for dropout in DROPOUTS:
                    config_name = f"{model_name}_{embed_name}_len{max_len}_drop{dropout}"
                    print(f"\n--- Starting 3-Fold CV for {config_name} ---")

                    dataset = TextDataset(texts, labels, vocab, max_len)
                    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
                    fold_f1_scores = []

                    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
                        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
                        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

                        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
                        val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_subsampler)

                        model = BaseHateModel(model_name, vocab_size, embed_matrix, dropout).to(DEVICE)
                        criterion = nn.BCELoss()
                        optimizer = optim.Adam(model.parameters(), lr=0.001)

                        # Training Loop
                        model.train()
                        for epoch in range(EPOCHS):
                            for batch_texts, batch_labels in train_loader:
                                batch_texts, batch_labels = batch_texts.to(DEVICE), batch_labels.to(DEVICE)
                                optimizer.zero_grad()
                                predictions = model(batch_texts)
                                loss = criterion(predictions, batch_labels)
                                loss.backward()
                                optimizer.step()

                        # Validation Loop
                        model.eval()
                        all_preds, all_targets = [], []
                        with torch.no_grad():
                            for batch_texts, batch_labels in val_loader:
                                batch_texts = batch_texts.to(DEVICE)
                                predictions = model(batch_texts).cpu().numpy()
                                preds_binary = (predictions > 0.5).astype(int)
                                all_preds.extend(preds_binary)
                                all_targets.extend(batch_labels.numpy())

                        fold_f1 = f1_score(all_targets, all_preds)
                        fold_f1_scores.append(fold_f1)
                        print(f"Fold {fold + 1} F1-Score: {fold_f1:.4f}")

                    avg_f1 = np.mean(fold_f1_scores)
                    print(f"Average CV F1-Score for {config_name}: {avg_f1:.4f}")
                    results.append(
                        {'Model': model_name, 'Embedding': embed_name, 'Max_Len': max_len, 'Dropout': dropout,
                         'CV_F1_Score': avg_f1})

    results_df = pd.DataFrame(results).sort_values(by='CV_F1_Score', ascending=False)
    print("\n" + "=" * 50)
    print("DEEP LEARNING GRID SEARCH COMPLETE")
    print("=" * 50)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(MODEL_DIR, 'dl_cv_results.csv'), index=False)


if __name__ == "__main__":
    train_and_evaluate()