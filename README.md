# Beyond Literal Meaning

## The Effect of Prompt Engineering on Sentiment Classification in Movie Reviews

**Author:** Clara Maltez Mouro Rodrigues  
**Student ID:** V14244  
**Course:** Natural Language Processing  
**University:** Università degli Studi di Milano  
**Academic Year:** 2026–2027  
**Track 3:** Interventions and Causality

## 1. Project Overview

This project investigates the effect of explicit linguistic instructions on sentiment classification using a Large Language Model (LLM).

The main research question is:

> Does explicitly instructing a large language model to consider negation, irony, and sarcasm improve its sentiment classification performance on movie reviews?

The study adopts a controlled paired experimental design to estimate the effect of a prompt-level intervention.

## 2. Dataset

The project uses the IMDb Dataset of 50K Movie Reviews, which contains 50,000 English-language movie reviews labelled as positive or negative.

Dataset source: [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

After preprocessing:

- Total reviews: 49,578
- Training set: 39,662 reviews
- Test set: 9,916 reviews
- Experimental sample: 400 reviews
- Positive experimental reviews: 200
- Negative experimental reviews: 200

## 3. Methodology

### Traditional Baseline

- TF-IDF representation
- Unigram features
- Logistic Regression
- Stratified train-test split

### Large Language Model

The experiment uses Qwen2.5-3B-Instruct, executed locally through Ollama.

Two prompting conditions are compared:

- **Control:** Standard sentiment classification instructions.
- **Treatment:** Additional instructions to consider negation, irony, and sarcasm.

Each review is evaluated under both conditions, resulting in 800 predictions. The review text, language model and inference configuration remain constant across conditions.

### Statistical Analysis

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Average Treatment Effect (ATE)
- Stratified paired bootstrap (10,000 resamples)
- 95% confidence interval
- Exact McNemar test
- Linguistic subgroup analysis
- Qualitative error analysis

## 4. Main Results

| Model | Accuracy |
|---|---:|
| TF-IDF + Logistic Regression | 90.25% |
| LLM - Control | 93.50% |
| LLM - Treatment | 92.25% |

All three models and conditions are evaluated on the same experimental sample of 400 reviews.

### Treatment Effect

| Statistic | Result |
|---|---:|
| Both conditions correct | 367 |
| Both conditions incorrect | 24 |
| Improved predictions | 2 |
| Worsened predictions | 7 |
| Average Treatment Effect | -1.25 percentage points |
| 95% Confidence Interval | [-2.75, 0.00] percentage points |
| McNemar p-value | 0.179688 |

Although the treatment condition achieved slightly lower accuracy, the difference was not statistically significant at the 5% significance level.

The results do not provide statistically significant evidence that explicit linguistic guidance improves sentiment classification under the tested conditions.

## 5. Project Structure

```
NLP-Final-Project/
├── data/
├── results/
├── main.py
├── llm_experiment.py
├── run_experiment.py
├── statistical_analysis.py
├── error_analysis.py
├── results_visualization.py
├── requirements.txt
└── README.md
```

## 6. Installation

Python 3.13 was used for this project.

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

Install Ollama:

https://ollama.com/download

Download the language model:

```bash
ollama pull qwen2.5:3b
```

Ensure Ollama is running before executing the LLM experiment.

Download the IMDb dataset and place the CSV file inside the data directory as:

```
data/IMDB Dataset.csv
```

The linguistic annotation spreadsheet must also be available at:

```
data/human_validation_20.xlsx
```

## 7. How to Run

Execute the following scripts from the project root directory.

### Step 1 - Preprocessing and Traditional Baseline

```bash
python main.py
```

### Step 2 - Pilot Experiment (Optional)

```bash
python llm_experiment.py
```

### Step 3 - LLM Experiment

```bash
python run_experiment.py
```

### Step 4 - Statistical Analysis

```bash
python statistical_analysis.py
```

### Step 5 - Error and Linguistic Analysis

```bash
python error_analysis.py
```

### Step 6 - Results Visualization

```bash
python results_visualization.py
```

The generated results and figures are stored in the results directory.

The LLM experiment supports resuming from previously saved predictions. To repeat the experiment from scratch, back up and remove the existing llm_predictions.csv file from the results directory.

## 8. Limitations

- Evaluation restricted to English-language movie reviews.
- A single large language model and inference seed.
- Limited experimental sample size.
- Combined linguistic intervention, preventing the identification of individual effects.
- Small linguistic annotation subset.
- AI-assisted annotations reviewed by a single annotator.
- Absence of inter-annotator agreement measurements.
- Possible exposure to IMDb reviews during LLM pretraining.
- Full input retention within the configured context window was not independently verified.

The experiment measures observable model behaviour rather than internal model mechanisms.

## 9. Reference Paper

Zhang et al. (2024).

*Sentiment Analysis in the Era of Large Language Models: A Reality Check.*

Findings of NAACL 2024.

## 10. AI Usage Disclosure

OpenAI ChatGPT (GPT-6) was used to support project development, experimental design, Python implementation, statistical methodology, linguistic annotation assistance, and scientific writing.

The linguistic annotations were reviewed by the author. The sentiment reference labels originate from the IMDb dataset, while the experimental predictions were generated by the local Qwen model.