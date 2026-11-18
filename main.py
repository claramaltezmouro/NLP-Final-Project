import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. Load the dataset
df = pd.read_csv("data/IMDB Dataset.csv")

# 2. Data exploration
print("First five rows:")
print(df.head())

print("\nDataset dimensions:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicated reviews:")
print(df["review"].duplicated().sum())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

# 3. Create the results directory
Path("results").mkdir(exist_ok=True)

# 4. Sentiment distribution visualization
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))

ax = sns.countplot(
    data=df,
    x="sentiment",
    hue="sentiment",
    order=["positive", "negative"],
    palette={
        "positive": "#81c784",
        "negative": "#e57373"
    },
    legend=False
)

# Add numerical labels
for container in ax.containers:
    ax.bar_label(container, padding=5, fmt="%d")

plt.title("Distribution of Movie Review Sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.ylim(0, df["sentiment"].value_counts().max() * 1.15)

plt.tight_layout()

# 5. Save and display the figure
plt.savefig("results/sentiment_distribution.png", dpi=300)
plt.show()

import re
import html

# 6. Text preprocessing
def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df = df.dropna(subset=["review", "sentiment"]).copy()
df["clean_review"] = df["review"].apply(clean_text)

# Remove empty and duplicated reviews
df = df[df["clean_review"] != ""]
df = df.drop_duplicates(subset=["clean_review"])
df = df.reset_index(drop=True)

# 7. Numerical encoding
df["label"] = df["sentiment"].map({
    "negative": 0,
    "positive": 1
})

# 8. Review length analysis
df["word_count"] = df["clean_review"].str.split().str.len()

print("\n--- PREPROCESSED DATASET ---")
print("Remaining reviews:", len(df))

print("\nSentiment distribution after cleaning:")
print(df["sentiment"].value_counts())

print("\nReview length statistics:")
print(df.groupby("sentiment")["word_count"].describe())

# 9. Review length visualization
plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="word_count",
    hue="sentiment",
    palette={
        "positive": "#81c784",
        "negative": "#e57373"
    },
    bins=80,
    element="step",
    stat="density",
    common_norm=False,
    alpha=0.35
)

plt.title("Review Length Distribution by Sentiment")
plt.xlabel("Number of Words")
plt.ylabel("Density")
plt.xlim(0, df["word_count"].quantile(0.99))
plt.tight_layout()
plt.savefig("results/review_length_distribution.png", dpi=300)
plt.show()

# 10. Save descriptive statistics
df.groupby("sentiment")["word_count"].describe().to_csv(
    "results/review_length_statistics.csv"
)

print("\nPreprocessing completed successfully.")

# 11. Train-test split
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

X = df["clean_review"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n--- TRAIN-TEST SPLIT ---")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 12. TF-IDF representation (unigrams)
tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 1),
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 13. Logistic Regression
model_tfidf = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model_tfidf.fit(X_train_tfidf, y_train)

# 14. Model evaluation
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)

print("\n--- MODEL 1: TF-IDF UNIGRAMS ---")
print("Accuracy:", accuracy_score(y_test, y_pred_tfidf))

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred_tfidf,
    target_names=["Negative", "Positive"]
))

# 15. Confusion matrix
cm = confusion_matrix(y_test, y_pred_tfidf)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.title("Confusion Matrix - TF-IDF Unigrams")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.tight_layout()
plt.savefig("results/confusion_matrix_unigrams.png", dpi=300)
plt.close()

# 16. Save evaluation results
report = classification_report(
    y_test,
    y_pred_tfidf,
    target_names=["Negative", "Positive"],
    output_dict=True
)

pd.DataFrame(report).transpose().to_csv(
    "results/metrics_unigrams.csv"
)

print("\nModel 1 completed successfully.")


# ==========================================================
# 17. EXPERIMENTAL SAMPLE SELECTION
# ==========================================================

RANDOM_SEED = 42
SAMPLES_PER_CLASS = 200

test_df = pd.DataFrame({
    "review": X_test,
    "label": y_test,
    "tfidf_prediction": y_pred_tfidf
}).reset_index(drop=True)

positive_sample = test_df[test_df["label"] == 1].sample(
    n=SAMPLES_PER_CLASS,
    random_state=RANDOM_SEED
)

negative_sample = test_df[test_df["label"] == 0].sample(
    n=SAMPLES_PER_CLASS,
    random_state=RANDOM_SEED
)

experiment_df = pd.concat(
    [positive_sample, negative_sample],
    ignore_index=True
)

experiment_df = experiment_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

experiment_df.insert(
    0, "review_id",
    np.arange(1, len(experiment_df) + 1)
)

experiment_df["sentiment"] = experiment_df["label"].map({
    0: "negative",
    1: "positive"
})

print("\n--- EXPERIMENTAL SAMPLE ---")
print("Total reviews:", len(experiment_df))
print(experiment_df["sentiment"].value_counts())

# ==========================================================
# 18. TF-IDF BASELINE ON THE EXPERIMENTAL SAMPLE
# ==========================================================

sample_accuracy = accuracy_score(
    experiment_df["label"],
    experiment_df["tfidf_prediction"]
)

print("\n--- EXPERIMENTAL SAMPLE BASELINE ---")
print(f"Accuracy: {sample_accuracy:.4f}")

print(classification_report(
    experiment_df["label"],
    experiment_df["tfidf_prediction"],
    target_names=["Negative", "Positive"]
))

sample_report = classification_report(
    experiment_df["label"],
    experiment_df["tfidf_prediction"],
    target_names=["Negative", "Positive"],
    output_dict=True
)

pd.DataFrame(sample_report).transpose().to_csv(
    "results/metrics_experimental_baseline.csv"
)

# ==========================================================
# 19. SAVE EXPERIMENTAL AND ANNOTATION DATASETS
# ==========================================================

experiment_df.to_csv(
    "data/experimental_sample.csv",
    index=False
)

annotation_df = experiment_df[
    ["review_id", "review"]
].copy()

annotation_df["negation"] = ""
annotation_df["irony"] = ""
annotation_df["sarcasm"] = ""
annotation_df["notes"] = ""

annotation_df.to_csv(
    "data/linguistic_annotations.csv",
    index=False
)

print("\nExperimental sample saved successfully.")