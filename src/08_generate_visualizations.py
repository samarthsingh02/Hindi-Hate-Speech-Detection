import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

# Create directory for saving graphs
OUTPUT_DIR = '../reports/figures/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set global visual style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})


def plot_model_progression():
    """Graph 1: The steady climb of F1-Scores across different architectures."""
    models = [
        "SVM",
        "Bi-LSTM",
        "mBERT",
        "XLM-R",
        "MuRIL",
        "Soft Ens.",
        "Weighted Ens."
    ]
    f1_scores = [0.7193, 0.7259, 0.7514, 0.7603, 0.7620, 0.7688, 0.7701]

    plt.figure(figsize=(10, 6))
    colors = ['#cccccc', '#cccccc', '#9ecae1', '#4292c6', '#2171b5', '#08519c', '#08306b']

    bars = plt.bar(models, f1_scores, color=colors)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.002, f'{yval:.4f}', ha='center', va='bottom',
                 fontweight='bold')

    plt.ylim(0.70, 0.78)
    plt.title('Performance Progression Across Architectures (F1-Score)')
    plt.ylabel('CV F1-Score')
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_model_progression.png'), dpi=300)
    plt.close()
    print("Saved: 01_model_progression.png")


def plot_precision_vs_recall():
    """Graph 2: Precision vs Recall across all models."""
    models = ["SVM", "Bi-LSTM", "mBERT", "XLM-R", "MuRIL", "Soft Ens.", "Weighted Ens."]
    precision = [0.7493, 0.7409, 0.7914, 0.7803, 0.7920, 0.8007, 0.7595]
    recall = [0.6916, 0.7115, 0.7152, 0.7413, 0.7342, 0.7393, 0.7809]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width / 2, precision, width, label='Precision (Accuracy of Flags)', color='#4c72b0')
    bars2 = plt.bar(x + width / 2, recall, width, label='Recall (Hate Speech Caught)', color='#dd8452')

    plt.ylim(0.65, 0.85)
    plt.title('Precision vs. Recall Tradeoff Across Architectures')
    plt.ylabel('Score')
    plt.xticks(x, models, rotation=30, ha='right')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_precision_vs_recall.png'), dpi=300)
    plt.close()
    print("Saved: 02_precision_vs_recall.png")


def plot_confusion_matrix():
    """Graph 3: Confusion Matrix for the final Weighted Ensemble."""
    # Based on 31,038 validation samples and the 4,031 errors identified
    # Actual Normal (0) = ~21,727 | Actual Hate (1) = ~9,311
    # False Positives = 1,668 | False Negatives = 2,363
    # True Negatives = 21,727 - 1,668 = 20,059
    # True Positives = 9,311 - 2,363 = 6,948

    cm = np.array([[20059, 1668],
                   [2363, 6948]])

    plt.figure(figsize=(7, 6))

    # Custom color map to match standard confusion matrix styling
    cmap = LinearSegmentedColormap.from_list('custom_blue', ['#f7fbff', '#08306b'])

    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=['Normal (0)', 'Hate (1)'],
                yticklabels=['Normal (0)', 'Hate (1)'],
                annot_kws={"size": 14, "weight": "bold"})

    plt.title('Confusion Matrix: Final Weighted Ensemble', pad=15)
    plt.ylabel('Actual Label (Ground Truth)', fontweight='bold')
    plt.xlabel('Predicted Label (Ensemble Output)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_confusion_matrix.png'), dpi=300)
    plt.close()
    print("Saved: 03_confusion_matrix.png")


def plot_error_distribution():
    """Graph 4: Breakdown of exactly how the model failed (FP vs FN)."""
    labels = ['False Positives\n(Hallucinated Hate)', 'False Negatives\n(Missed Hate)']
    sizes = [1668, 2363]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.05, 0)

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=False, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})

    plt.title('Error Type Distribution (Total Errors: 4,031)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_error_distribution.png'), dpi=300)
    plt.close()
    print("Saved: 04_error_distribution.png")


def plot_disagreement_analysis():
    """Graph 5: MuRIL vs XLM-R Disagreements."""
    labels = ['Models Agreed on Error', 'Models Disagreed']
    sizes = [2896, 1135]
    colors = ['#99ff99', '#ffcc99']
    explode = (0, 0.1)

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=False, startangle=140, textprops={'fontsize': 12, 'fontweight': 'bold'})

    plt.title('Model Disagreement During Failure')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_model_disagreement.png'), dpi=300)
    plt.close()
    print("Saved: 05_model_disagreement.png")


if __name__ == "__main__":
    print("Generating project graphs...")
    plot_model_progression()
    plot_precision_vs_recall()
    plot_confusion_matrix()
    plot_error_distribution()
    plot_disagreement_analysis()
    print("All graphs successfully saved to reports/figures/")