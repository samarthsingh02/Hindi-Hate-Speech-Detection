import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# --- Configuration & Paths ---
DATA_PATH = '../data/processed/unified_dataset.csv'
MODEL_DIR = '../models/'


def main():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH, encoding='utf-8')
    except FileNotFoundError:
        print("Error: Unified dataset not found.")
        return

    # Drop any rows that became NaN or empty after text cleaning
    df.dropna(subset=['text', 'label'], inplace=True)
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.strip() != '']  # Remove any purely blank strings

    # Train-Test Split (Hold-out set for final evaluation after CV)
    print("Splitting data (80% Train/CV, 20% Unseen Test)...")
    X = df['text'].astype(str)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 1. Define Feature Extraction Strategies
    # Note: max_features is set to prevent RAM exhaustion on large datasets.
    feature_dict = {
        'Word_1_1': TfidfVectorizer(analyzer='word', ngram_range=(1, 1), max_features=20000),

        'Word_1_2': TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=30000),

        'Char_3_5': TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=30000),

        'Combined_Word_Char': FeatureUnion([
            ('word_1_2', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=15000)),
            ('char_3_5', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=15000))
        ])
    }

    # 2. Define Models and Hyperparameter Grid
    # Both use L2 regularization by default. class_weight='balanced' addresses the 70:30 split.
    models_dict = {
        'Logistic_Regression': (
            LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42),
            {'model__C': [0.01, 0.1, 1, 10]}
        ),
        'Linear_SVM': (
            LinearSVC(class_weight='balanced', max_iter=2000, random_state=42, dual='auto'),
            {'model__C': [0.01, 0.1, 1, 10]}
        )
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    results_log = []

    # 3. Execute Grid Search Pipeline
    for feature_name, vectorizer in feature_dict.items():
        print(f"\n{'=' * 50}")
        print(f"STARTING FEATURE SET: {feature_name}")
        print(f"{'=' * 50}")

        for model_name, (model_instance, param_grid) in models_dict.items():
            print(f"\nRunning 5-Fold CV for: {model_name} with {feature_name} features...")

            # Create a pipeline combining the specific vectorizer and model
            pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('model', model_instance)
            ])

            # Configure GridSearch (optimizing for F1-score on the positive 'Hate' class)
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,  # Use all available CPU cores to speed up execution
                verbose=1
            )

            # Fit GridSearch on the training data
            grid_search.fit(X_train, y_train)

            print(f"\nBest Parameters for {model_name} ({feature_name}): {grid_search.best_params_}")
            print(f"Best Cross-Validation F1-Score: {grid_search.best_score_:.4f}")

            # 4. Final Evaluation on Unseen Test Data
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            print(f"\nTest Set Performance for {model_name} ({feature_name}):")
            report = classification_report(y_test, y_pred, target_names=['Non-Hate (0)', 'Hate (1)'])
            print(report)

            # 5. Save the best version of this specific model/feature combination
            save_name = f"{model_name}_{feature_name}_best.pkl"
            joblib.dump(best_model, os.path.join(MODEL_DIR, save_name))

            results_log.append({
                'Feature_Set': feature_name,
                'Model': model_name,
                'Best_C': grid_search.best_params_['model__C'],
                'CV_F1_Score': grid_search.best_score_
            })

    # Summary Output
    print("\n" + "#" * 50)
    print("EXPERIMENTAL GRID SEARCH COMPLETE")
    print("#" * 50)
    results_df = pd.DataFrame(results_log)
    print(results_df.sort_values(by='CV_F1_Score', ascending=False).to_string(index=False))
    results_df.to_csv(os.path.join(MODEL_DIR, 'traditional_cv_results_summary.csv'), index=False)
    print("\nSummary saved to models directory.")


if __name__ == "__main__":
    main()