import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Load statistical results

summary = pd.read_csv("results/statistical_summary.csv").iloc[0]

Path("results").mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# 2. Classification accuracy comparison

models = [
    "TF-IDF + LR",
    "LLM Control",
    "LLM Treatment"
]

accuracies = [
    90.25,
    summary["control_accuracy"] * 100,
    summary["treatment_accuracy"] * 100
]

fig, ax = plt.subplots(figsize=(7, 4.5))

bars = ax.bar(
    models,
    accuracies,
    color=["#A6A6A6", "#C65B91", "#EBA8C8"],
    width=0.6
)

ax.set_ylim(0, 105)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Sentiment Classification Performance")

for bar, value in zip(bars, accuracies):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig("results/accuracy_comparison.png", dpi=300)
plt.close()

# 3. Paired classification outcomes

categories = [
    "Both correct",
    "Both incorrect",
    "Improved",
    "Worsened"
]

counts = [
    summary["both_correct"],
    summary["both_wrong"],
    summary["improved"],
    summary["worsened"]
]

fig, ax = plt.subplots(figsize=(7, 4.5))

bars = ax.barh(
    categories,
    counts,
    color=["#C65B91", "#A6A6A6", "#70AD85", "#D66B6B"]
)

ax.invert_yaxis()
ax.set_xlim(0, 410)
ax.set_xlabel("Number of reviews")
ax.set_title("Paired Classification Outcomes")

for bar, value in zip(bars, counts):
    ax.text(
        value + 5,
        bar.get_y() + bar.get_height() / 2,
        str(int(value)),
        va="center"
    )

plt.tight_layout()
plt.savefig("results/paired_outcomes.png", dpi=300)
plt.close()

# 4. Treatment effect and confidence interval

ate = summary["average_treatment_effect"] * 100
lower = summary["ci_lower"] * 100
upper = summary["ci_upper"] * 100

fig, ax = plt.subplots(figsize=(7, 3))

ax.errorbar(
    ate,
    0,
    xerr=[[ate - lower], [upper - ate]],
    fmt="o",
    color="#C65B91",
    capsize=7,
    markersize=8,
    linewidth=2
)

ax.axvline(0, color="gray", linestyle="--")
ax.set_xlim(min(-4, lower - 0.5), max(2, upper + 0.5))
ax.set_yticks([])
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("Treatment effect (percentage points)")
ax.set_title("Average Treatment Effect with 95% CI")

ax.text(
    ate,
    0.15,
    f"{ate:.2f} pp [{lower:.2f}, {upper:.2f}]",
    ha="center"
)

plt.tight_layout()
plt.savefig("results/treatment_effect.png", dpi=300)
plt.close()

print("All experimental figures generated successfully.")

# 5. Accuracy by sentiment class

sentiment = pd.read_csv(
    "results/sentiment_class_analysis.csv"
)

fig, ax = plt.subplots(figsize=(7, 4.5))

x = range(len(sentiment))
width = 0.35

bars1 = ax.bar(
    [i - width / 2 for i in x],
    sentiment["control_accuracy"] * 100,
    width,
    label="Control",
    color="#C65B91"
)

bars2 = ax.bar(
    [i + width / 2 for i in x],
    sentiment["treatment_accuracy"] * 100,
    width,
    label="Treatment",
    color="#EBA8C8"
)

ax.set_xticks(list(x))
ax.set_xticklabels(sentiment["sentiment"].str.capitalize())
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 115)
ax.set_title("Classification Accuracy by Sentiment")
ax.legend()

ax.bar_label(bars1, fmt="%.1f%%", padding=3)
ax.bar_label(bars2, fmt="%.1f%%", padding=3)

plt.tight_layout()
plt.savefig(
    "results/accuracy_by_sentiment.png",
    dpi=300
)
plt.close()

print("Sentiment class figure generated successfully.")

# 6. Linguistic phenomena comparison

linguistic = pd.read_csv(
    "results/linguistic_phenomena_analysis.csv"
)

present = linguistic[
    linguistic["presence"] == "Present"
]

fig, ax = plt.subplots(figsize=(7, 4.5))

x = range(len(present))
width = 0.35

bars1 = ax.bar(
    [i - width / 2 for i in x],
    present["control_accuracy"] * 100,
    width,
    label="Control",
    color="#C65B91"
)

bars2 = ax.bar(
    [i + width / 2 for i in x],
    present["treatment_accuracy"] * 100,
    width,
    label="Treatment",
    color="#EBA8C8"
)

labels = [
    f"{row['phenomenon'].capitalize()}\n"
    f"(n = {row['number_of_reviews']})"
    for _, row in present.iterrows()
]

ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 120)
ax.set_title("Accuracy by Linguistic Phenomenon")
ax.legend()

ax.bar_label(bars1, fmt="%.1f%%", padding=3)
ax.bar_label(bars2, fmt="%.1f%%", padding=3)

plt.tight_layout()

plt.savefig(
    "results/linguistic_phenomena.png",
    dpi=300
)

plt.close()

print("Linguistic phenomena figure generated successfully.")