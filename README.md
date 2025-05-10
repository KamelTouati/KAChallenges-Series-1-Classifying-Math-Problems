```markdown
# KAChallenges Series 1: Classifying Math Problems

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition%20Page-blue.svg)](https://www.kaggle.com/competitions/classification-of-math-problems-by-kasut-academy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Optional: Add a LICENSE file if you wish -->

This repository contains the solution and code for the **KAChallenges Series 1: Classifying Math Problems** competition hosted by KAUST Academy on Kaggle. The goal was to classify math problems written in natural language into 8 distinct topics. The evaluation metric was F1-micro.

**My Final F1-micro Score (LLM Ensemble): 0.8365 (Rank ~122)**

## Table of Contents
1.  [Competition Overview](#competition-overview)
2.  [My Approach](#my-approach)
    *   [1. Classical Machine Learning Pipeline](#1-classical-machine-learning-pipeline)
    *   [2. Transformer-based LLM Ensemble (Best Performing)](#2-transformer-based-llm-ensemble-best-performing)
3.  [Key Techniques & Learnings](#key-techniques--learnings)
4.  [Results Summary](#results-summary)
5.  [Acknowledgements](#acknowledgements)


## Competition Overview

The challenge involved classifying mathematics problems, provided as natural language text, into one of eight predefined categories:
0.  Algebra
1.  Geometry and Trigonometry
2.  Calculus and Analysis
3.  Probability and Statistics
4.  Number Theory
5.  Combinatorics and Discrete Math
6.  Linear Algebra
7.  Abstract Algebra and Topology

Submissions were evaluated using the F1 Score with micro averaging (F1-micro).

## My Approach

I developed and experimented with two primary strategies, detailed in the respective Jupyter Notebooks:

### 1. Classical Machine Learning Pipeline

*   **Notebook:** `kachallenges-series-1-classifying-math-problems_classical-ml.ipynb` (Please verify and update this filename if different)
*   **Preprocessing:**
    *   Extensive text cleaning: lowercasing, converting mathematical symbols (e.g., '∫', '∑', '√') to their textual equivalents ('integral', 'sum', 'square root').
    *   Normalization: Using regex for handling superscripts/subscripts.
    *   Tokenization, careful stopword removal (preserving key math-related terms), and lemmatization using NLTK.
*   **Feature Engineering:**
    *   **TF-IDF Vectorization:** Using `TfidfVectorizer` with n-grams (1 to 3) and optimized feature counts.
    *   **Custom Features:** A rich set of hand-crafted features (details in the notebook).
*   **Modeling & Ensemble:**
    *   Experimented with Logistic Regression, Random Forest, XGBoost, and LightGBM.
    *   Final model was a **majority-voting ensemble** of these individual classifiers.
*   **Result:** Achieved an **F1-micro score of 0.7736** with this approach.

### 2. Transformer-based LLM Ensemble (Best Performing)

*   **Notebook:** `kachallenges-series-1-classifying-math-problems_llm-ensemble.ipynb` (Please verify and update this filename if different)
*   This approach leveraged the contextual understanding capabilities of pre-trained language models.
*   **Models:**
    *   Fine-tuned an ensemble of three powerful pre-trained language models:
        1.  `allenai/scibert_scivocab_uncased` (specialized for scientific text)
        2.  `roberta-base`
        3.  `bert-base-uncased`
*   **Training & Ensembling Strategy:**
    *   Implemented a **k-fold cross-validation strategy** (3 folds used).
    *   Each model was fine-tuned on the training data folds.
    *   For the final ensemble prediction, the **probabilities from each fine-tuned model across the folds were averaged**.
*   **Result:** This LLM ensemble was my top-performing solution, achieving an **F1-micro score of 0.8365** on the leaderboard.

## Key Techniques & Learnings

*   **LLM Superiority:** Fine-tuned Transformer models demonstrated superior performance.
*   **Ensembling Power:** Both approaches benefited, with the LLM ensemble excelling.
*   **K-Fold Cross-Validation:** Crucial for robust LLM fine-tuning.
*   **Strategic Preprocessing:** Important even for classical ML.
*   **Iterative Development:** Key to achieving final results.
*   **Resource Management:** Essential for LLM fine-tuning.

## Results Summary

| Approach                        | F1-micro Score |
| :------------------------------ | :------------- |
| Classical ML Ensemble           | 0.7736         |
| **Transformer LLM Ensemble**    | **0.8365**     |

## Acknowledgements

*   **KAUST Academy** and **Kaggle** for hosting this competition.
*   The developers of `scikit-learn`, `pandas`, `numpy`, `nltk`, `transformers`, `PyTorch`, and other open-source libraries used.
```

