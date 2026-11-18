import pandas as pd
import numpy as np
from scipy.stats import binomtest

# 1. Load experimental results

results = pd.read_csv("results/llm_predictions.csv")
sample = pd.read_csv("data/experimental_sample.csv")

assert len(results) == 800, "Experiment not completed."
assert not results.duplicated(["review_id", "condition"]).any()

# 2. Classification accuracy

accuracy = results.groupby("condition")["correct"].mean()

control_accuracy = accuracy["Control"]
treatment_accuracy = accuracy["Treatment"]

print("\n--- CLASSIFICATION ACCURACY ---")
print(f"Control: {control_accuracy:.2%}")
print(f"Treatment: {treatment_accuracy:.2%}")

# 3. Paired outcome analysis

paired = results.pivot(
    index="review_id",
    columns="condition",
    values="correct"
).astype(int)

assert len(paired) == 400

paired["difference"] = paired["Treatment"] - paired["Control"]

both_correct = (
    (paired["Control"] == 1) & (paired["Treatment"] == 1)
).sum()

both_wrong = (
    (paired["Control"] == 0) & (paired["Treatment"] == 0)
).sum()

improved = (paired["difference"] == 1).sum()
worsened = (paired["difference"] == -1).sum()

print("\n--- PAIRED OUTCOMES ---")
print("Both correct:", both_correct)
print("Both incorrect:", both_wrong)
print("Improved:", improved)
print("Worsened:", worsened)

# 4. Average Treatment Effect (ATE)

ate = paired["difference"].mean()

print("\n--- TREATMENT EFFECT ---")
print(f"ATE: {ate * 100:.2f} percentage points")

# 5. Stratified paired bootstrap confidence interval

paired = paired.join(
    sample.set_index("review_id")["label"]
)

rng = np.random.default_rng(42)
n_bootstrap = 10000
bootstrap_effects = []

for _ in range(n_bootstrap):

    sampled_groups = []

    for label, group in paired.groupby("label"):

        differences = group["difference"].to_numpy()

        resampled = rng.choice(
            differences,
            size=len(differences),
            replace=True
        )

        sampled_groups.append(resampled)

    bootstrap_effects.append(
        np.concatenate(sampled_groups).mean()
    )

ci_lower, ci_upper = np.percentile(
    bootstrap_effects, [2.5, 97.5]
)

print("\n--- 95% CONFIDENCE INTERVAL ---")
print(
    f"[{ci_lower * 100:.2f}, {ci_upper * 100:.2f}] "
    "percentage points"
)

# 6. Exact McNemar test

discordant = improved + worsened

if discordant > 0:

    p_value = binomtest(
        int(improved),
        int(discordant),
        p=0.5,
        alternative="two-sided"
    ).pvalue

else:
    p_value = 1.0

alpha = 0.05

print("\n--- MCNEMAR TEST ---")
print(f"p-value: {p_value:.6f}")

if p_value < alpha:
    print("Statistically significant difference.")
else:
    print("No statistically significant difference.")

# 7. Save statistical results

summary = pd.DataFrame([{
    "control_accuracy": control_accuracy,
    "treatment_accuracy": treatment_accuracy,
    "average_treatment_effect": ate,
    "ci_lower": ci_lower,
    "ci_upper": ci_upper,
    "both_correct": both_correct,
    "both_wrong": both_wrong,
    "improved": improved,
    "worsened": worsened,
    "p_value": p_value
}])

summary.to_csv(
    "results/statistical_summary.csv",
    index=False
)

paired.to_csv("results/paired_outcomes.csv")

print("\nStatistical analysis completed successfully.")

# 8. Final evaluation metrics

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

evaluation_results = []

for condition in ["Control", "Treatment"]:

    subset = results[results["condition"] == condition]

    y_true = subset["actual"]
    y_pred = subset["prediction"]

    valid = y_pred.isin(["positive", "negative"])

    evaluation_results.append({
        "condition": condition,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred,
            labels=["negative", "positive"],
            average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred,
            labels=["negative", "positive"],
            average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred,
            labels=["negative", "positive"],
            average="macro", zero_division=0
        ),
        "valid_predictions": valid.sum(),
        "invalid_predictions": (~valid).sum()
    })

evaluation = pd.DataFrame(evaluation_results)

print("\n--- FINAL EVALUATION METRICS ---")
print(evaluation.to_string(index=False))

evaluation.to_csv(
    "results/final_evaluation_metrics.csv",
    index=False
)