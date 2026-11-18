
import ollama
import time

# ==========================================================
# 1. MODEL CONFIGURATION
# ==========================================================

MODEL_NAME = "qwen2.5:3b"
RANDOM_SEED = 42

# ==========================================================
# 2. EXPERIMENTAL PROMPTS
# ==========================================================

CONTROL_PROMPT = (
    "Classify the overall sentiment of the following "
    "movie review as positive or negative. "
    "Respond with only one word: positive or negative."
)

TREATMENT_PROMPT = (
    "Classify the overall sentiment of the following "
    "movie review as positive or negative. "
    "Pay particular attention to negation, irony, "
    "and sarcasm. "
    "Respond with only one word: positive or negative."
)

# ==========================================================
# 3. CLASSIFICATION FUNCTION
# ==========================================================

def classify_review(review, instruction):

    start_time = time.perf_counter()

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": instruction + "\n\n" + review
            }
        ],
        options={
            "temperature": 0,
            "seed": RANDOM_SEED,
            "num_ctx": 8192,
            "num_predict": 16
        }
    )

    prediction = response.message.content.strip().lower()
    elapsed_time = time.perf_counter() - start_time

    return prediction, elapsed_time


# ==========================================================
# 4. PRELIMINARY FEASIBILITY TEST
# ==========================================================

if __name__ == "__main__":

    example_review = (
        "What a wonderful movie! "
        "I've never been so bored in my life."
    )

    print("--- PRELIMINARY LLM TEST ---")
    print("Model:", MODEL_NAME)
    print("Review:", example_review)
    print("Expected sentiment: negative")

    for condition, prompt in [
        ("Control", CONTROL_PROMPT),
        ("Treatment", TREATMENT_PROMPT)
    ]:

        prediction, elapsed = classify_review(
            example_review, prompt
        )

        print(f"\nCondition: {condition}")
        print(f"Prediction: {prediction}")
        print(f"Execution time: {elapsed:.2f} seconds")


# ==========================================================
# 5. PILOT EXPERIMENT
# ==========================================================

import pandas as pd
import html
import re
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    df = pd.read_csv("data/IMDB Dataset.csv")

    def clean_text(text):
        text = html.unescape(text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["clean_review"] = df["review"].apply(clean_text)
    df = df[df["clean_review"] != ""]
    df = df.drop_duplicates(subset=["clean_review"])
    df = df.reset_index(drop=True)

    df["label"] = df["sentiment"].map({
        "negative": 0,
        "positive": 1
    })

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    pilot_df = train_df.sample(
        n=10,
        random_state=42
    )

    print("\n--- PILOT EXPERIMENT ---")

    pilot_results = []

    for i, (_, row) in enumerate(pilot_df.iterrows(), 1):

        for condition, prompt in [
            ("Control", CONTROL_PROMPT),
            ("Treatment", TREATMENT_PROMPT)
        ]:

            prediction, elapsed = classify_review(
                row["clean_review"], prompt
            )

            pilot_results.append({
                "review_id": i,
                "condition": condition,
                "actual": row["sentiment"],
                "prediction": prediction,
                "correct": prediction == row["sentiment"],
                "valid": prediction in ["positive", "negative"],
                "runtime_seconds": elapsed
            })

            print(
                f"Review {i}/10 | {condition} | "
                f"Actual: {row['sentiment']} | "
                f"Predicted: {prediction} | "
                f"Time: {elapsed:.2f}s"
            )

    pilot_results = pd.DataFrame(pilot_results)

    pilot_results.to_csv(
        "results/pilot_results.csv",
        index=False
    )

    average_time = pilot_results["runtime_seconds"].mean()
    estimated_minutes = 800 * average_time / 60

    print("\n--- PILOT SUMMARY ---")
    print("Total predictions:", len(pilot_results))
    print("Valid predictions:", pilot_results["valid"].sum())
    print(f"Average time: {average_time:.2f} seconds")
    print(f"Estimated full runtime: {estimated_minutes:.2f} minutes")
