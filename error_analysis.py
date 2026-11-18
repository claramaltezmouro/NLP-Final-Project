import pandas as pd

# 1. Load experimental data

sample = pd.read_csv("data/experimental_sample.csv")
predictions = pd.read_csv("results/llm_predictions.csv")
paired = pd.read_csv("results/paired_outcomes.csv")

# 2. Merge predictions and original reviews

predictions_wide = predictions.pivot(
    index="review_id",
    columns="condition",
    values="prediction"
).reset_index()

predictions_wide = predictions_wide.rename(columns={
    "Control": "control_prediction",
    "Treatment": "treatment_prediction"
})

analysis = sample.merge(
    predictions_wide, on="review_id"
)

analysis = analysis.merge(
    paired[["review_id", "difference", "Control", "Treatment"]],
    on="review_id"
)

# 3. Categorise experimental outcomes

def categorise_outcome(row):

    if row["difference"] == 1:
        return "Improved"

    if row["difference"] == -1:
        return "Worsened"

    if row["Control"] == 1:
        return "Both correct"

    return "Both incorrect"


analysis["outcome"] = analysis.apply(
    categorise_outcome, axis=1
)

# 4. Extract relevant cases

improved = analysis[analysis["outcome"] == "Improved"]

worsened = analysis[analysis["outcome"] == "Worsened"]

both_wrong = analysis[analysis["outcome"] == "Both incorrect"]

# 5. Save results

analysis.to_csv(
    "results/detailed_error_analysis.csv", index=False
)

improved.to_csv(
    "results/improved_reviews.csv", index=False
)

worsened.to_csv(
    "results/worsened_reviews.csv", index=False
)

both_wrong.to_csv(
    "results/both_incorrect_reviews.csv", index=False
)

# 6. Display summary

print("\n--- ERROR ANALYSIS ---")

print(analysis["outcome"].value_counts())

print("\n--- IMPROVED REVIEWS ---")

for _, row in improved.iterrows():

    print("\nReview ID:", row["review_id"])
    print("Actual sentiment:", row["sentiment"])
    print("Control:", row["control_prediction"])
    print("Treatment:", row["treatment_prediction"])
    print("Review:", row["review"])

print("\n--- WORSENED REVIEWS ---")

for _, row in worsened.iterrows():

    print("\nReview ID:", row["review_id"])
    print("Actual sentiment:", row["sentiment"])
    print("Control:", row["control_prediction"])
    print("Treatment:", row["treatment_prediction"])
    print("Review:", row["review"])

print("\nError analysis completed successfully.")

# 7. Treatment effect by sentiment class

sentiment_analysis = analysis.groupby("sentiment").agg(
    number_of_reviews=("review_id", "count"),
    control_accuracy=("Control", "mean"),
    treatment_accuracy=("Treatment", "mean"),
    treatment_effect=("difference", "mean")
)

print("\n--- SENTIMENT CLASS ANALYSIS ---")
print(sentiment_analysis.to_string())

sentiment_analysis.to_csv(
    "results/sentiment_class_analysis.csv"
)

# 8. Linguistic phenomena analysis

annotations = pd.read_excel(
    "data/human_validation_20.xlsx"
)

linguistic_analysis = analysis.merge(
    annotations[["review_id", "negation",
                 "irony", "sarcasm"]],
    on="review_id",
    validate="one_to_one"
)

assert len(linguistic_analysis) == 20

phenomena = ["negation", "irony", "sarcasm"]

linguistic_results = []

for phenomenon in phenomena:

    for presence in [0, 1]:

        subset = linguistic_analysis[
            linguistic_analysis[phenomenon] == presence
        ]

        linguistic_results.append({
            "phenomenon": phenomenon,
            "presence": "Present" if presence == 1 else "Absent",
            "number_of_reviews": len(subset),
            "control_accuracy": subset["Control"].mean(),
            "treatment_accuracy": subset["Treatment"].mean(),
            "treatment_effect": subset["difference"].mean(),
            "improved": (subset["difference"] == 1).sum(),
            "worsened": (subset["difference"] == -1).sum()
        })

linguistic_summary = pd.DataFrame(linguistic_results)

print("\n--- LINGUISTIC PHENOMENA ANALYSIS ---")
print(linguistic_summary.to_string(index=False))

linguistic_summary.to_csv(
    "results/linguistic_phenomena_analysis.csv",
    index=False
)

linguistic_analysis.to_csv(
    "results/annotated_review_outcomes.csv",
    index=False
)

print("\nLinguistic analysis completed successfully.")