
import pandas as pd
import random
import csv
from pathlib import Path
from llm_experiment import (
    classify_review,
    CONTROL_PROMPT,
    TREATMENT_PROMPT,
    RANDOM_SEED
)

# ==========================================================
# 1. EXPERIMENTAL CONFIGURATION
# ==========================================================

INPUT_FILE = Path("data/experimental_sample.csv")
OUTPUT_FILE = Path("results/llm_predictions.csv")

rng = random.Random(RANDOM_SEED)

PROMPTS = {
    "Control": CONTROL_PROMPT,
    "Treatment": TREATMENT_PROMPT
}

# ==========================================================
# 2. LOAD EXPERIMENTAL SAMPLE
# ==========================================================

df = pd.read_csv(INPUT_FILE)

assert len(df) == 400
assert df["review_id"].is_unique
assert df["label"].value_counts().to_dict() == {
    0: 200, 1: 200
}

# ==========================================================
# 3. CHECKPOINT AND RESUME
# ==========================================================

OUTPUT_FILE.parent.mkdir(exist_ok=True)

completed = set()

if OUTPUT_FILE.exists():
    previous = pd.read_csv(OUTPUT_FILE)

    completed = set(zip(
        previous["review_id"],
        previous["condition"]
    ))

columns = [
    "review_id",
    "condition",
    "actual",
    "prediction",
    "valid",
    "correct",
    "runtime_seconds"
]

total = len(df) * 2
progress = len(completed)

# ==========================================================
# 4. PAIRED EXPERIMENT
# ==========================================================

print("--- MAIN LLM EXPERIMENT ---")
print("Reviews:", len(df))
print("Total classifications:", total)
print("Already completed:", progress)

with open(
    OUTPUT_FILE,
    "a",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(file, fieldnames=columns)

    if file.tell() == 0:
        writer.writeheader()

    for _, row in df.iterrows():

        review_id = int(row["review_id"])
        actual = row["sentiment"]

        # Randomize condition order within each pair
        conditions = ["Control", "Treatment"]
        rng.shuffle(conditions)

        for condition in conditions:

            if (review_id, condition) in completed:
                continue

            prediction, elapsed = classify_review(
                row["review"],
                PROMPTS[condition]
            )

            valid = prediction in [
                "positive", "negative"
            ]

            correct = valid and prediction == actual

            writer.writerow({
                "review_id": review_id,
                "condition": condition,
                "actual": actual,
                "prediction": prediction,
                "valid": valid,
                "correct": correct,
                "runtime_seconds": elapsed
            })

            file.flush()

            progress += 1

            print(
                f"[{progress}/{total}] | "
                f"Review {review_id} | "
                f"{condition} | "
                f"Prediction: {prediction} | "
                f"Time: {elapsed:.2f}s",
                flush=True
            )

# ==========================================================
# 5. SAVE EXPERIMENTAL SUMMARY
# ==========================================================

results = pd.read_csv(OUTPUT_FILE)

assert len(results) == total
assert not results.duplicated(
    ["review_id", "condition"]
).any()

summary = results.groupby("condition").agg(
    predictions=("review_id", "count"),
    accuracy=("correct", "mean"),
    valid_predictions=("valid", "sum"),
    mean_runtime=("runtime_seconds", "mean"),
    total_runtime=("runtime_seconds", "sum")
)

summary.to_csv("results/llm_summary.csv")

print("\n--- EXPERIMENTAL SUMMARY ---")
print(summary)

print("\nExperiment completed successfully.")
