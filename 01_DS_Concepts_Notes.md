# Data Science — Important Concepts (Round 1 Notes)

> How to study: for every concept be able to say **What it is → Intuition → Formula (if any) → When to use → Pros/Cons**.

## Table of Contents
1. [Statistics — Descriptive](#1-statistics--descriptive)
2. [Probability](#2-probability)
3. [Probability Distributions](#3-probability-distributions)
4. [Inferential Statistics & Hypothesis Testing](#4-inferential-statistics--hypothesis-testing)
5. [A/B Testing & Experimentation](#5-ab-testing--experimentation)
6. [Machine Learning Fundamentals](#6-machine-learning-fundamentals)
7. [Regression Models](#7-regression-models)
8. [Classification Models](#8-classification-models)
9. [Tree-Based & Ensemble Models](#9-tree-based--ensemble-models)
10. [Unsupervised Learning](#10-unsupervised-learning)
11. [Model Evaluation Metrics](#11-model-evaluation-metrics)
12. [Data Preprocessing & Feature Engineering](#12-data-preprocessing--feature-engineering)
13. [Model Validation, Tuning & Overfitting](#13-model-validation-tuning--overfitting)
14. [Deep Learning](#14-deep-learning)
15. [NLP & Generative AI](#15-nlp--generative-ai)
16. [Time Series](#16-time-series)
17. [Recommendation Systems](#17-recommendation-systems)
18. [SQL for Data Science](#18-sql-for-data-science)
19. [Python / Pandas / NumPy](#19-python--pandas--numpy)
20. [MLOps & Deployment Basics](#20-mlops--deployment-basics)
21. [Linear Algebra & Calculus Essentials](#21-linear-algebra--calculus-essentials)
22. [Quick Revision Cheat Sheet](#22-quick-revision-cheat-sheet)

---

## 1. Statistics — Descriptive

### Measures of central tendency
| Measure | Definition | Use when |
|---|---|---|
| **Mean** | Sum / n | Symmetric data, no outliers |
| **Median** | Middle value of sorted data | Skewed data / outliers (e.g., income) |
| **Mode** | Most frequent value | Categorical data |

- Right (positive) skew: **Mean > Median > Mode** (long tail on right, e.g., salaries).
- Left (negative) skew: **Mean < Median < Mode**.

### Measures of spread
- **Range** = max − min
- **Variance** σ² = Σ(x − μ)² / n  (sample: divide by **n − 1** → Bessel's correction, makes it unbiased)
- **Standard deviation** σ = √variance (same unit as data)
- **IQR** = Q3 − Q1 (robust to outliers)
- **Coefficient of variation** = σ / μ (compare variability across different scales)

### Outlier detection
- **IQR rule**: outlier if < Q1 − 1.5·IQR or > Q3 + 1.5·IQR
- **Z-score**: |z| > 3, where z = (x − μ)/σ
- Visual: box plot, scatter plot
- Model based: Isolation Forest, DBSCAN, Local Outlier Factor

### Shape
- **Skewness**: asymmetry. 0 = symmetric.
- **Kurtosis**: tailedness. Normal = 3 (excess kurtosis 0). High kurtosis → heavy tails → more extreme values.

### Relationship between variables
- **Covariance**: direction of linear relationship; unit-dependent; range (−∞, ∞).
- **Pearson correlation** r = Cov(X,Y)/(σx·σy): strength & direction of *linear* relationship, range [−1, 1].
- **Spearman correlation**: rank-based; captures *monotonic* relationships; robust to outliers.
- **Correlation ≠ causation** – confounders, reverse causality, coincidence.

### Percentiles & quantiles
- Pth percentile: value below which P% of data lies. Median = 50th percentile.

---

## 2. Probability

### Basic rules
- P(A ∪ B) = P(A) + P(B) − P(A ∩ B)
- Independent: P(A ∩ B) = P(A)·P(B)
- Conditional: P(A|B) = P(A ∩ B) / P(B)
- Mutually exclusive: P(A ∩ B) = 0 (mutually exclusive events with non-zero prob are **not** independent)
- Complement: P(A') = 1 − P(A) → useful for "at least one" problems: P(at least one) = 1 − P(none)

### Bayes' Theorem
```
P(A|B) = P(B|A) · P(A) / P(B)
P(B) = P(B|A)P(A) + P(B|A')P(A')     (law of total probability)
```
**Classic example** – disease test: prevalence 1%, sensitivity 99%, false positive rate 5%.
P(disease | +) = 0.99·0.01 / (0.99·0.01 + 0.05·0.99) ≈ **16.7%** → base rate matters!

### Counting
- Permutations (order matters): nPr = n! / (n−r)!
- Combinations (order doesn't matter): nCr = n! / (r!(n−r)!)

### Expected value & variance
- E[X] = Σ x·P(x); E[aX + b] = aE[X] + b; E[X+Y] = E[X] + E[Y] (always)
- Var(aX + b) = a²Var(X); Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y)

### Key theorems
- **Law of Large Numbers**: sample mean → population mean as n grows.
- **Central Limit Theorem (CLT)**: distribution of the *sample mean* approaches Normal(μ, σ²/n) as n grows (n ≥ 30 rule of thumb), **regardless of the population distribution** (given finite variance). This is why z/t-tests work.

---

## 3. Probability Distributions

| Distribution | Type | Models | Mean | Variance |
|---|---|---|---|---|
| **Bernoulli(p)** | Discrete | Single yes/no trial | p | p(1−p) |
| **Binomial(n,p)** | Discrete | # successes in n trials | np | np(1−p) |
| **Poisson(λ)** | Discrete | # events in fixed interval (calls/hr) | λ | λ |
| **Geometric(p)** | Discrete | # trials until 1st success | 1/p | (1−p)/p² |
| **Uniform(a,b)** | Continuous | Equally likely values | (a+b)/2 | (b−a)²/12 |
| **Normal(μ,σ²)** | Continuous | Heights, errors, averages | μ | σ² |
| **Exponential(λ)** | Continuous | Time between events (memoryless) | 1/λ | 1/λ² |
| **Log-normal** | Continuous | Income, prices (log is normal) | — | — |
| **t-distribution** | Continuous | Small-sample means; heavier tails | 0 | >1 |
| **Chi-square** | Continuous | Sum of squared normals; variance tests, independence | k | 2k |

**Empirical rule (Normal)**: 68% within 1σ, 95% within 2σ (exactly 1.96σ), 99.7% within 3σ.

**Testing normality**: Q-Q plot, histogram, Shapiro-Wilk test, Kolmogorov-Smirnov test.

---

## 4. Inferential Statistics & Hypothesis Testing

### Population vs Sample
- **Parameter** describes population (μ, σ); **statistic** describes sample (x̄, s).
- **Standard error** SE = σ / √n — spread of the sampling distribution of the mean.

### Sampling methods
- Simple random, stratified (divide into groups, sample each), cluster, systematic, convenience (biased).
- **Biases**: selection bias, survivorship bias, response/non-response bias, sampling bias.

### Hypothesis testing steps
1. State **H₀** (null: no effect) and **H₁** (alternative).
2. Choose significance level **α** (usually 0.05).
3. Choose test & compute test statistic.
4. Compute **p-value**.
5. If p < α → reject H₀; else fail to reject H₀ (never "accept H₀").

### p-value
> Probability of observing data **at least as extreme** as what we saw, **assuming H₀ is true**.
- It is NOT the probability that H₀ is true.

### Errors
| | H₀ true | H₀ false |
|---|---|---|
| Reject H₀ | **Type I error (α)** – false positive | Correct (Power = 1−β) |
| Fail to reject | Correct | **Type II error (β)** – false negative |

- **Power** = 1 − β = probability of detecting a real effect. Typically target 80%.
- Power increases with: larger sample size, larger effect size, larger α, lower variance.

### Confidence interval
- 95% CI = x̄ ± 1.96 · SE
- Interpretation: if we repeat the sampling many times, 95% of such intervals will contain the true parameter. (Not "95% probability the parameter lies in this interval" — that's the Bayesian credible interval.)

### Which test to use?
| Situation | Test |
|---|---|
| Compare mean to known value, σ known / large n | **z-test** |
| Compare mean, σ unknown / small n | **one-sample t-test** |
| Compare means of 2 independent groups | **two-sample (independent) t-test** (Welch's if unequal variance) |
| Same subjects before/after | **paired t-test** |
| Compare means of 3+ groups | **ANOVA** (then Tukey post-hoc) |
| Two categorical variables independent? | **Chi-square test of independence** |
| Observed vs expected frequencies | **Chi-square goodness-of-fit** |
| Compare proportions (conversion rates) | **two-proportion z-test** |
| Non-normal data, 2 independent groups | **Mann-Whitney U** |
| Non-normal paired data | **Wilcoxon signed-rank** |
| Non-normal 3+ groups | **Kruskal-Wallis** |
| Compare variances | **F-test / Levene's test** |

### One-tailed vs two-tailed
- Two-tailed: H₁: μ ≠ μ₀ (any difference). One-tailed: H₁: μ > μ₀ (direction specified in advance).

### Multiple testing problem
- Running 20 tests at α=0.05 → ~64% chance of at least one false positive (1 − 0.95²⁰).
- Corrections: **Bonferroni** (α/m, conservative), **Holm**, **Benjamini-Hochberg** (controls False Discovery Rate).

### Parametric vs non-parametric
- Parametric assumes a distribution (normality) → more power when assumptions hold.
- Non-parametric: fewer assumptions, rank-based, robust.

### Bayesian vs Frequentist
- Frequentist: parameters are fixed, data random; uses p-values, CIs.
- Bayesian: parameters have distributions; prior + likelihood → posterior. Gives "probability B is better than A".

---

## 5. A/B Testing & Experimentation

### End-to-end A/B test
1. **Define the goal & hypothesis** – e.g., "New checkout button increases conversion".
2. **Choose metrics**
   - **Primary (OEC)**: conversion rate
   - **Secondary**: revenue per user, AOV
   - **Guardrail metrics**: latency, crash rate, unsubscribe rate (must not get worse)
3. **Randomization unit** – usually user (not session/page view) to keep experience consistent.
4. **Sample size** (per group, proportions):
   ```
   n ≈ 16 · σ² / δ²      (rule of thumb for α=0.05, power=0.8)
   σ² = p(1−p) for proportions, δ = minimum detectable effect (absolute)
   ```
   Example: baseline 10%, MDE 1% absolute → n ≈ 16·0.09/0.0001 = 14,400 per group.
5. **Duration** – at least 1–2 full weeks to capture weekly seasonality; n / daily traffic.
6. **Run** – no peeking (or use sequential testing).
7. **Sanity checks** – **Sample Ratio Mismatch (SRM)** check with chi-square; A/A test.
8. **Analyze** – two-proportion z-test / t-test, CI, practical significance vs statistical significance.
9. **Decide** – launch, iterate, or kill; consider cost, long-term effects.

### Common pitfalls
| Pitfall | Explanation | Fix |
|---|---|---|
| **Peeking** | Stopping as soon as p<0.05 inflates false positives | Fixed horizon or sequential testing (alpha spending) |
| **Novelty / primacy effect** | Users react to change temporarily | Run longer, analyze by cohort over time |
| **Network effects / interference** | Treatment affects control (social networks, marketplaces) | Cluster randomization, geo-experiments, switchback tests |
| **SRM** | Unequal split vs designed → bug in assignment | Chi-square check; don't trust results |
| **Multiple metrics / variants** | False positives | Bonferroni/BH correction |
| **Simpson's paradox** | Trend reverses when data combined | Stratify, check segments |
| **Seasonality / holidays** | Non-representative period | Run full weeks, avoid unusual periods |
| **Low power** | Can't detect real effect | Bigger sample, variance reduction (**CUPED**) |

### Other experimentation concepts
- **CUPED**: uses pre-experiment data as covariate to reduce variance → smaller sample needed.
- **Multi-armed bandits**: dynamically allocate more traffic to the winning variant (explore vs exploit) — good for short-lived decisions (e.g., headlines).
- **Quasi-experiments** when A/B is impossible: Difference-in-Differences, Regression Discontinuity, Propensity Score Matching, Synthetic Control, Instrumental Variables.
- **Statistical vs practical significance**: a 0.01% lift might be significant with huge n but not worth engineering cost.

---

## 6. Machine Learning Fundamentals

### Types of learning
- **Supervised**: labeled data → regression (continuous) / classification (categorical).
- **Unsupervised**: no labels → clustering, dimensionality reduction, anomaly detection.
- **Semi-supervised**: few labels + lots of unlabeled.
- **Self-supervised**: labels generated from data itself (e.g., next-word prediction).
- **Reinforcement learning**: agent learns via rewards (states, actions, policy).

### Parametric vs non-parametric models
- Parametric: fixed number of parameters (linear/logistic regression). Fast, strong assumptions.
- Non-parametric: complexity grows with data (KNN, decision trees). Flexible, need more data.

### Generative vs Discriminative
- Discriminative models P(y|x) directly — logistic regression, SVM, NN.
- Generative models P(x|y)P(y) / joint — Naive Bayes, GMM, GANs, LLMs.

### Bias–Variance Trade-off
```
Expected error = Bias² + Variance + Irreducible noise
```
| | High bias (underfitting) | High variance (overfitting) |
|---|---|---|
| Train error | High | Low |
| Test error | High | High |
| Fix | More complex model, more features, less regularization | More data, regularization, simpler model, dropout, bagging, early stopping |

### Gradient Descent
- Update: θ = θ − η · ∇L(θ), η = learning rate.
- **Batch GD**: whole dataset per step (stable, slow).
- **Stochastic GD**: one sample (noisy, fast, can escape local minima).
- **Mini-batch GD**: compromise (standard in DL).
- Learning rate too high → diverges; too low → slow.

### Loss functions
| Task | Loss |
|---|---|
| Regression | MSE (penalizes large errors), MAE (robust to outliers), Huber (mix) |
| Binary classification | Binary cross-entropy / log loss |
| Multi-class | Categorical cross-entropy |
| SVM | Hinge loss |

### Curse of dimensionality
As dimensions increase, data becomes sparse, distances become less meaningful, more data needed → use feature selection, PCA, regularization.

### No Free Lunch theorem
No single model is best for all problems → try multiple, validate.

---

## 7. Regression Models

### Linear Regression
- y = β₀ + β₁x₁ + … + βₙxₙ + ε
- Fitted by **Ordinary Least Squares** (minimize sum of squared residuals). Closed form: β = (XᵀX)⁻¹Xᵀy
- **Assumptions (LINE + no multicollinearity)**:
  1. **L**inearity between X and y
  2. **I**ndependence of errors (no autocorrelation — Durbin-Watson test)
  3. **N**ormality of residuals (Q-Q plot)
  4. **E**qual variance — homoscedasticity (residual vs fitted plot)
  5. No (perfect) **multicollinearity** (check **VIF** > 5–10 is a problem)
- Coefficient interpretation: one-unit increase in x₁ → β₁ change in y, holding others constant.

### Evaluation
- **R²** = 1 − SS_res/SS_tot → proportion of variance explained.
- **Adjusted R²** penalizes adding useless features.
- MAE, MSE, RMSE, MAPE.

### Regularization
| | Ridge (L2) | Lasso (L1) | Elastic Net |
|---|---|---|---|
| Penalty | λΣβ² | λΣ\|β\| | mix of both |
| Effect | Shrinks coefficients toward 0 | Shrinks some exactly to 0 → **feature selection** | Both |
| Use | Multicollinearity, many small effects | Sparse models, many irrelevant features | Correlated features + selection |

**Why L1 gives sparsity**: the L1 constraint region is a diamond with corners on the axes; the loss contours tend to touch it at a corner, where some coefficients are exactly 0.

### Other regression
- **Polynomial regression** – add x², x³ terms (still linear in parameters).
- **Quantile regression** – predicts median/percentiles.
- **Poisson regression** – count data.

---

## 8. Classification Models

### Logistic Regression
- p = σ(z) = 1 / (1 + e^(−z)), z = β₀ + βᵀx
- Models **log-odds**: log(p/(1−p)) = z → linear decision boundary.
- Loss: log loss (cross-entropy), trained by gradient descent / MLE.
- Coefficient: e^β = **odds ratio** for one-unit increase.
- Why not MSE? MSE with sigmoid is non-convex → many local minima.
- Multi-class: one-vs-rest or **softmax** (multinomial).

### K-Nearest Neighbors (KNN)
- Classify by majority vote of k nearest points (Euclidean, Manhattan, cosine).
- Lazy learner — no training, slow prediction. **Needs feature scaling.**
- Small k → overfitting; large k → underfitting. Choose k via CV (odd k for binary).

### Naive Bayes
- Applies Bayes' theorem assuming features are **conditionally independent** given class.
- Types: Gaussian (continuous), Multinomial (word counts), Bernoulli (binary).
- Fast, works well for text/spam. **Laplace smoothing** handles zero probabilities.

### Support Vector Machine (SVM)
- Finds hyperplane maximizing the **margin** between classes; **support vectors** are points on the margin.
- **C**: regularization — high C = hard margin (overfit risk), low C = softer margin.
- **Kernel trick**: implicitly maps data to higher dimension (linear, polynomial, **RBF**). γ in RBF controls reach of each point.
- Good for high-dimensional data; slow on large datasets; needs scaling.

---

## 9. Tree-Based & Ensemble Models

### Decision Tree
- Recursively split on feature that best separates the data.
- Split criteria:
  - **Gini impurity** = 1 − Σpᵢ²
  - **Entropy** = −Σpᵢ log₂ pᵢ; **Information Gain** = parent entropy − weighted child entropy
  - Regression: variance reduction / MSE
- Pros: interpretable, handles non-linearity, no scaling needed, handles mixed types.
- Cons: **overfits easily**, unstable (high variance).
- Control: max_depth, min_samples_split, min_samples_leaf, pruning.

### Ensemble methods
| | **Bagging** | **Boosting** | **Stacking** |
|---|---|---|---|
| How | Train models in parallel on bootstrap samples, average/vote | Train sequentially; each model fixes errors of previous | Meta-model learns how to combine base models |
| Reduces | **Variance** | **Bias** (and variance) | Both |
| Example | Random Forest | AdaBoost, GBM, XGBoost, LightGBM, CatBoost | — |

### Random Forest
- Bagging of decision trees + **random subset of features at each split** (decorrelates trees).
- **OOB (out-of-bag) error**: ~37% of samples not in each bootstrap → free validation.
- Feature importance: mean decrease in impurity or permutation importance.
- Robust, little tuning, handles high dimensions.

### Gradient Boosting
- Each new tree fits the **residuals (negative gradient)** of the current ensemble.
- Prediction = sum of trees × learning rate.
- **XGBoost**: regularization (L1/L2), handles missing values, parallel split finding, pruning.
- **LightGBM**: leaf-wise growth, histogram-based → very fast on big data.
- **CatBoost**: native categorical handling, ordered boosting.
- Key hyperparameters: n_estimators, learning_rate, max_depth, subsample, colsample_bytree.
- AdaBoost: increases weights of misclassified samples.

---

## 10. Unsupervised Learning

### K-Means
1. Pick k centroids randomly (better: **k-means++**).
2. Assign each point to nearest centroid.
3. Recompute centroids as mean.
4. Repeat until convergence.
- Minimizes within-cluster sum of squares (inertia).
- Choose k: **Elbow method**, **Silhouette score** (−1 to 1, higher better), Gap statistic.
- Limitations: assumes spherical, similar-size clusters; sensitive to outliers & initialization; needs scaling.

### Hierarchical clustering
- Agglomerative (bottom-up) / divisive. Visualized with **dendrogram**.
- Linkage: single, complete, average, **Ward**. No need to pre-specify k.

### DBSCAN
- Density-based: parameters **eps** (radius) and **minPts**.
- Finds arbitrary-shaped clusters, marks outliers as noise, no need for k.
- Struggles with varying densities.

### Gaussian Mixture Models
- Soft clustering (probability of belonging to each cluster), fitted with **EM algorithm**.

### PCA (Principal Component Analysis)
- Finds orthogonal directions of **maximum variance**.
- Steps: standardize → covariance matrix → eigenvectors/eigenvalues → sort → project.
- Explained variance ratio → choose components covering ~90–95% variance.
- Linear, components lose interpretability; **must scale data first**.
- Others: **t-SNE**, **UMAP** (non-linear, for visualization), LDA (supervised, maximizes class separation), SVD.

### Anomaly detection
- Statistical (z-score, IQR), Isolation Forest, One-Class SVM, Autoencoders (high reconstruction error = anomaly).

### Association rules (Market basket)
- **Support** = P(A∩B), **Confidence** = P(B|A), **Lift** = Confidence / P(B) (>1 = positive association). Apriori algorithm.

---

## 11. Model Evaluation Metrics

### Confusion matrix
| | Predicted + | Predicted − |
|---|---|---|
| **Actual +** | TP | FN (Type II) |
| **Actual −** | FP (Type I) | TN |

| Metric | Formula | Focus |
|---|---|---|
| **Accuracy** | (TP+TN)/All | Misleading for imbalanced data |
| **Precision** | TP/(TP+FP) | "Of predicted positives, how many correct?" — use when **FP is costly** (spam filter) |
| **Recall / Sensitivity / TPR** | TP/(TP+FN) | "Of actual positives, how many found?" — use when **FN is costly** (cancer, fraud) |
| **Specificity / TNR** | TN/(TN+FP) | Correct negatives |
| **F1-score** | 2PR/(P+R) | Harmonic mean, balance P & R |
| **Fβ** | β>1 weights recall more | |
| **FPR** | FP/(FP+TN) | x-axis of ROC |

### ROC-AUC vs PR-AUC
- **ROC curve**: TPR vs FPR across thresholds. **AUC** = probability model ranks a random positive above a random negative. 0.5 = random, 1 = perfect.
- **PR curve**: precision vs recall — **better for highly imbalanced** data (ROC can look optimistic).
- **Threshold tuning**: default 0.5 is arbitrary; choose based on business cost.

### Other classification metrics
- **Log loss** (probability quality), **Brier score**, **Cohen's kappa**, **MCC** (good for imbalance).
- **Calibration**: are predicted probabilities true probabilities? (reliability plot, Platt scaling, isotonic regression).
- **Lift / Gain charts** – marketing use.
- Multi-class: macro avg (equal weight classes), micro avg (equal weight samples), weighted avg.

### Regression metrics
| Metric | Notes |
|---|---|
| MAE | Average absolute error, robust to outliers, interpretable |
| MSE | Penalizes large errors heavily |
| RMSE | Same unit as target |
| MAPE | Percentage error; breaks when actual ≈ 0 |
| R² / Adjusted R² | Variance explained |

### Ranking metrics
- Precision@K, Recall@K, **MAP**, **NDCG** (position-aware), MRR, Hit rate.

### Clustering metrics
- Silhouette score, Davies-Bouldin index, Calinski-Harabasz; with labels: ARI, NMI.

---

## 12. Data Preprocessing & Feature Engineering

### Missing values
- Types: **MCAR** (completely at random), **MAR** (depends on observed data), **MNAR** (depends on missing value itself).
- Handling: drop rows/cols (if few), mean/median/mode imputation, KNN imputer, MICE (iterative), model-based, forward fill (time series), add **"is_missing" indicator** flag, tree models (XGBoost) handle natively.

### Encoding categorical variables
| Method | When |
|---|---|
| Label / Ordinal encoding | Ordinal categories (low < med < high); fine for trees |
| One-hot encoding | Nominal, low cardinality (beware dummy variable trap → drop one) |
| Target / mean encoding | High cardinality; **risk of leakage** → use CV/smoothing |
| Frequency encoding | High cardinality |
| Hashing | Very high cardinality |
| Embeddings | Deep learning |

### Feature scaling
- **Standardization** (z-score): (x − μ)/σ → mean 0, std 1. For linear models, SVM, PCA, NN, KNN.
- **Min-Max normalization**: (x − min)/(max − min) → [0,1]. Sensitive to outliers. For NN, images.
- **Robust scaler**: uses median & IQR → outlier-robust.
- **Trees don't need scaling.**
- Fit scaler **only on training data**, then transform test.

### Transformations
- Log / Box-Cox / Yeo-Johnson for skewed data; binning; polynomial features; interaction features.

### Feature engineering ideas
- Date: day of week, month, is_weekend, holiday, time since last event.
- Aggregations: user's avg spend, count of transactions in last 7/30 days (RFM: Recency, Frequency, Monetary).
- Ratios: price per sqft, clicks/impressions.
- Text: length, TF-IDF, embeddings.
- Domain knowledge features are usually the most valuable.

### Feature selection
- **Filter**: correlation, chi-square, mutual information, variance threshold.
- **Wrapper**: RFE (recursive feature elimination), forward/backward selection.
- **Embedded**: Lasso, tree feature importance.
- Also: SHAP values, permutation importance.

### Imbalanced data
- Use right metrics (PR-AUC, F1, recall), not accuracy.
- **Resampling**: random undersampling, oversampling, **SMOTE** (synthetic minority samples via interpolation), ADASYN. Apply **only on training set**.
- **Class weights** (`class_weight='balanced'`).
- Threshold tuning, anomaly detection approach, ensemble (BalancedRandomForest).

### Data leakage
> Information not available at prediction time sneaks into training → overly optimistic results.
- Examples: scaling before train-test split, target encoding without CV, features created after the event (e.g., "refund_issued" in fraud model), random split on time-series data, duplicate rows across train/test.
- Prevention: split first, use **Pipelines**, time-based splits, check suspiciously high accuracy.

### Multicollinearity
- Correlated predictors → unstable coefficients, hard interpretation (not a big issue for prediction).
- Detect: correlation matrix, **VIF**. Fix: drop features, PCA, Ridge.

---

## 13. Model Validation, Tuning & Overfitting

### Splits
- Train / Validation / Test (e.g., 70/15/15). Test set touched only once at the end.
- **Stratified split** for imbalanced classification.
- **Time-based split** for temporal data (train on past, test on future).

### Cross-validation
- **K-Fold**: split into k folds, train on k−1, validate on 1, repeat k times, average.
- **Stratified K-Fold**: preserves class ratio.
- **LOOCV**: k = n (expensive, low bias, high variance).
- **TimeSeriesSplit / walk-forward**: expanding window.
- **Group K-Fold**: same user/patient never in both train & val.

### Hyperparameter tuning
- **Grid search** (exhaustive), **Random search** (often better in high dims), **Bayesian optimization** (Optuna, Hyperopt), early stopping.
- Parameters are learned (weights); hyperparameters are set before training (learning rate, depth).

### Preventing overfitting
- More data, data augmentation
- Regularization (L1/L2), dropout, early stopping
- Simpler model / pruning / fewer features
- Cross-validation, ensembling (bagging)
- Batch normalization (slight regularization effect)

### Interpretability
- Global: feature importance, partial dependence plots (PDP), coefficients.
- Local: **SHAP** (Shapley values from game theory – fair contribution of each feature), **LIME** (local linear surrogate).

---

## 14. Deep Learning

### Neural network basics
- **Neuron**: output = activation(w·x + b).
- **Forward propagation**: compute predictions layer by layer.
- **Backpropagation**: compute gradients of loss w.r.t. weights using **chain rule**, then update with gradient descent.
- **Epoch**: one pass over entire data; **batch size**: samples per update; **iteration**: one update.

### Activation functions
| Function | Range | Notes |
|---|---|---|
| Sigmoid | (0,1) | Binary output; vanishing gradient; not zero-centered |
| Tanh | (−1,1) | Zero-centered; still vanishing gradient |
| **ReLU** | [0,∞) | Default for hidden layers; fast; "dying ReLU" problem |
| Leaky ReLU / ELU / GELU | | Fixes dying ReLU; GELU used in Transformers |
| Softmax | (0,1), sums to 1 | Multi-class output |

**Why non-linear activations?** Without them, stacked layers collapse into one linear function.

### Training problems
- **Vanishing gradients**: gradients shrink in deep nets (sigmoid/tanh) → ReLU, BatchNorm, residual connections, LSTM, good init (He/Xavier).
- **Exploding gradients**: gradient clipping, proper init.
- **Overfitting**: dropout, L2 (weight decay), early stopping, data augmentation.

### Optimizers
- SGD, SGD + **Momentum** (accelerates in consistent direction), **RMSprop** (adaptive LR per parameter), **Adam** (momentum + RMSprop; default choice), AdamW (decoupled weight decay).
- Learning-rate schedules: step decay, cosine annealing, warmup.

### Normalization
- **Batch Normalization**: normalizes activations per mini-batch → faster, stable training.
- **Layer Normalization**: per sample — used in Transformers/RNNs.

### CNN (Convolutional Neural Networks) – images
- **Convolution layer**: filters/kernels detect local patterns (edges → shapes → objects); parameter sharing.
- **Stride, padding**; output size = (W − K + 2P)/S + 1.
- **Pooling** (max/avg): downsampling, translation invariance.
- Architectures: LeNet, AlexNet, VGG, **ResNet** (skip connections), Inception, EfficientNet.
- **Transfer learning**: use pretrained model, freeze early layers, fine-tune the top.

### RNN / LSTM / GRU – sequences
- RNN: hidden state carries info across time steps; suffers vanishing gradients on long sequences.
- **LSTM**: cell state + gates (forget, input, output) → long-term memory.
- **GRU**: simpler (update & reset gates), fewer params.

### Transformers
- **Self-attention**: each token attends to all others: Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V.
- **Multi-head attention**, **positional encoding**, feed-forward layers, residual + LayerNorm.
- Parallelizable (unlike RNN), captures long-range dependencies.
- Encoder-only (**BERT** – understanding), decoder-only (**GPT** – generation), encoder-decoder (**T5**, translation).

### Other
- **Autoencoders**: encoder → bottleneck → decoder; dimensionality reduction, denoising, anomaly detection.
- **GANs**: generator vs discriminator adversarial training.
- **Embeddings**: dense vector representations of categorical items/words.

---

## 15. NLP & Generative AI

### Text preprocessing
- Lowercasing, tokenization, stop-word removal, **stemming** (crude chop: "running"→"run") vs **lemmatization** (dictionary root: "better"→"good"), removing punctuation.

### Text representation
- **Bag of Words**: word counts, ignores order.
- **TF-IDF**: TF × log(N/df) → down-weights common words.
- **N-grams**: capture local word order.
- **Word2Vec** (CBOW / Skip-gram), **GloVe**, FastText (subword) → dense semantic embeddings ("king − man + woman ≈ queen").
- **Contextual embeddings**: BERT — same word gets different vectors in different contexts.

### NLP tasks
- Sentiment analysis, NER, text classification, topic modeling (**LDA**), summarization, translation, Q&A.

### LLMs & GenAI (increasingly asked)
- **LLM**: large Transformer pretrained on next-token prediction, then instruction-tuned and aligned (RLHF / DPO).
- **Tokens, context window, temperature** (randomness), top-k / top-p sampling.
- **Prompt engineering**: zero-shot, few-shot, chain-of-thought.
- **RAG (Retrieval-Augmented Generation)**: embed documents → store in **vector DB** → retrieve relevant chunks by similarity → pass to LLM as context. Reduces hallucination, uses up-to-date/private data.
- **Fine-tuning**: full fine-tuning vs parameter-efficient (**LoRA**, QLoRA).
- **Hallucination**: confident but wrong output → RAG, grounding, evaluation.
- Evaluation: BLEU, ROUGE (summarization), perplexity, human eval, LLM-as-judge.
- Cosine similarity = A·B / (|A||B|) — used for embedding search.

---

## 16. Time Series

- **Components**: trend, seasonality, cyclic, noise (irregular).
- **Decomposition**: additive (y = T + S + R) vs multiplicative (y = T × S × R).
- **Stationarity**: constant mean, variance, autocorrelation over time. Test: **ADF** (H₀: non-stationary), KPSS. Fix: differencing, log transform.
- **ACF** (autocorrelation) → helps choose MA order q; **PACF** → AR order p.
- Models:
  - Moving average, exponential smoothing, **Holt-Winters** (trend + seasonality)
  - **ARIMA(p,d,q)**: AR + differencing + MA; **SARIMA** adds seasonality; ARIMAX adds external vars
  - **Prophet** (trend + seasonality + holidays)
  - ML: XGBoost with lag features, rolling means; DL: LSTM, Temporal Fusion Transformer
- **Validation**: never shuffle; walk-forward validation.
- Metrics: MAE, RMSE, MAPE, sMAPE, MASE.

---

## 17. Recommendation Systems

- **Popularity based**: simple baseline; solves cold start.
- **Content-based**: recommend items similar to what user liked (item features, TF-IDF, embeddings).
- **Collaborative filtering**:
  - User-based (similar users), item-based (similar items) using cosine/Pearson similarity.
  - **Matrix factorization** (SVD, ALS): user × item matrix ≈ user embeddings × item embeddings.
- **Hybrid** models; deep learning (two-tower models, neural CF).
- **Typical architecture**: candidate generation (retrieve ~1000s) → ranking (ML model ranks top 100) → re-ranking (diversity, business rules).
- **Cold start problem**: new user → popular items, onboarding questions, demographics; new item → content features.
- Implicit (clicks, views) vs explicit (ratings) feedback.
- Metrics: Precision@K, Recall@K, NDCG, MAP, coverage, diversity, novelty; online: CTR, conversion, watch time.

---

## 18. SQL for Data Science

### Execution order
```
FROM → JOIN → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT
```
(That's why you can't use a SELECT alias in WHERE.)

### Joins
- **INNER**: matching rows only; **LEFT**: all from left + matches; **RIGHT**; **FULL OUTER**; **CROSS** (Cartesian); **SELF JOIN**.
- Anti-join: `LEFT JOIN … WHERE b.id IS NULL` (rows in A not in B).

### WHERE vs HAVING
- WHERE filters rows before aggregation; HAVING filters groups after aggregation.

### Window functions (very frequently asked)
```sql
ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC)   -- 1,2,3,4
RANK()       -- 1,2,2,4 (gaps)
DENSE_RANK() -- 1,2,2,3 (no gaps)
LAG(col, 1) OVER (ORDER BY date)  -- previous row value
LEAD(col, 1)                      -- next row value
SUM(amount) OVER (PARTITION BY user ORDER BY date) -- running total
AVG(x) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) -- 7-day moving avg
NTILE(4) -- quartiles
```

### Classic queries
```sql
-- 2nd highest salary
SELECT MAX(salary) FROM emp WHERE salary < (SELECT MAX(salary) FROM emp);

-- Nth highest salary per department
WITH r AS (
  SELECT *, DENSE_RANK() OVER (PARTITION BY dept ORDER BY salary DESC) rnk FROM emp)
SELECT * FROM r WHERE rnk = 2;

-- Duplicates
SELECT email, COUNT(*) FROM users GROUP BY email HAVING COUNT(*) > 1;

-- Month-over-month growth
WITH m AS (SELECT DATE_TRUNC('month', order_date) mon, SUM(amount) rev FROM orders GROUP BY 1)
SELECT mon, rev, (rev - LAG(rev) OVER (ORDER BY mon)) * 1.0 / LAG(rev) OVER (ORDER BY mon) AS growth
FROM m;

-- Day-1 retention
SELECT COUNT(DISTINCT b.user_id) * 1.0 / COUNT(DISTINCT a.user_id) AS d1_retention
FROM signups a LEFT JOIN logins b
  ON a.user_id = b.user_id AND b.login_date = a.signup_date + INTERVAL '1 day';
```

### Other
- `UNION` (removes duplicates) vs `UNION ALL` (keeps, faster).
- `COUNT(*)` counts all rows; `COUNT(col)` ignores NULLs; `COUNT(DISTINCT col)`.
- `CASE WHEN` for conditional aggregation: `SUM(CASE WHEN status='done' THEN 1 ELSE 0 END)`.
- `COALESCE(x, 0)` to replace NULL. NULL = NULL is not true → use `IS NULL`.
- CTE (`WITH`) vs subquery — CTEs are more readable; indexes speed up lookups.
- `DELETE` (rows, can rollback) vs `TRUNCATE` (all rows, fast) vs `DROP` (table).

---

## 19. Python / Pandas / NumPy

### Python essentials
- **List** (mutable, ordered), **tuple** (immutable), **set** (unique, unordered), **dict** (key-value, hash map, O(1) lookup).
- List comprehension: `[x**2 for x in nums if x > 0]`
- `lambda`, `map`, `filter`, `zip`, `enumerate`.
- `*args` (variable positional), `**kwargs` (variable keyword).
- **Shallow vs deep copy** (`copy.copy` vs `copy.deepcopy`).
- **Generators** (`yield`) — lazy, memory efficient.
- **Decorators** — functions that wrap other functions.
- `is` (identity) vs `==` (equality).
- GIL — limits multi-threading for CPU-bound tasks → use multiprocessing.

### NumPy
- Vectorized operations are much faster than loops; **broadcasting** rules.
- `np.array`, `reshape`, `np.dot` / `@`, `axis=0` (down rows / per column), `axis=1` (across columns / per row).

### Pandas must-know
```python
df.head(), df.info(), df.describe(), df.shape, df.dtypes
df.isnull().sum()
df.fillna(df['col'].median()); df.dropna()
df.drop_duplicates()
df.loc[row_label, col_label]   # label based
df.iloc[0:5, 1:3]              # position based
df[df['age'] > 30]
df.groupby('city')['sales'].agg(['mean', 'sum', 'count'])
df.merge(df2, on='id', how='left')      # SQL-like join
pd.concat([df1, df2], axis=0)
df.pivot_table(values='sales', index='region', columns='month', aggfunc='sum')
df['col'].value_counts(normalize=True)
df['new'] = df['col'].apply(lambda x: x * 2)
df.sort_values('col', ascending=False)
df['rank'] = df.groupby('dept')['salary'].rank(ascending=False)
df['rolling'] = df['sales'].rolling(7).mean()
pd.to_datetime(df['date']); df['date'].dt.month
```
- `apply` is slow → prefer vectorized operations.
- `merge` (on columns) vs `join` (on index) vs `concat` (stack).

### Scikit-learn workflow
```python
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

pre = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)])
pipe = Pipeline([('pre', pre), ('model', LogisticRegression())])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pipe.fit(X_tr, y_tr)
```

---

## 20. MLOps & Deployment Basics

- **Model lifecycle**: problem → data → features → train → evaluate → deploy → monitor → retrain.
- **Deployment modes**: batch (nightly scores), real-time API (Flask/FastAPI + Docker), streaming, edge.
- **Tools**: Docker, Kubernetes, MLflow (tracking/registry), Airflow (pipelines), DVC (data versioning), cloud (AWS SageMaker, GCP Vertex AI, Azure ML).
- **Monitoring**:
  - **Data drift**: input distribution changes (PSI, KS test).
  - **Concept drift**: relationship between X and y changes (e.g., fraud patterns change).
  - Model performance, latency, errors.
- **Feature store**: consistent features for training & serving (avoid **training-serving skew**).
- **Shadow deployment, canary release, A/B test** new model vs old.
- Model versioning, reproducibility (random seeds, environment).

---

## 21. Linear Algebra & Calculus Essentials

- **Vectors, matrices, dot product** (similarity / projection), matrix multiplication (m×n · n×p = m×p).
- **Transpose, inverse** (only square, non-singular), **determinant** (0 → singular).
- **Rank**: number of linearly independent rows/columns.
- **Eigenvectors/eigenvalues**: Av = λv — direction unchanged by transformation; used in PCA.
- **SVD**: A = UΣVᵀ — used in PCA, recommender systems, compression.
- **Norms**: L1 = Σ|x|, L2 = √Σx².
- **Derivatives, partial derivatives, gradient** (vector of partials, points to steepest ascent), **chain rule** (basis of backprop).
- **Convex function**: any local minimum is global (linear/logistic regression loss is convex; NN loss is not).
- **MLE (Maximum Likelihood Estimation)**: choose parameters that maximize probability of observed data. MSE = MLE under Gaussian noise; log-loss = MLE under Bernoulli.
- **MAP**: MLE + prior → L2 regularization = Gaussian prior, L1 = Laplace prior.

---

## 22. Quick Revision Cheat Sheet

| Question | One-line answer |
|---|---|
| p-value | P(data this extreme \| H₀ true) |
| Type I / II | False positive / false negative |
| Power | P(reject H₀ \| H₁ true), 1−β, aim 0.8 |
| CLT | Sample means ~ Normal for large n regardless of population |
| Bias–variance | Simple = high bias; complex = high variance |
| L1 vs L2 | L1 → sparse (feature selection); L2 → shrink, handles multicollinearity |
| Bagging vs boosting | Parallel, ↓variance vs sequential, ↓bias |
| RF vs GBM | RF robust, less tuning; GBM usually more accurate, needs tuning |
| Precision vs recall | FP costly → precision; FN costly → recall |
| ROC vs PR AUC | Imbalanced → PR-AUC |
| Imbalanced data | Class weights, SMOTE (train only), threshold, right metric |
| Data leakage | Future/target info in training; split first, pipelines |
| Missing data | Drop, impute (median/KNN/MICE), indicator, tree-native |
| Scaling needed? | Yes: KNN, SVM, PCA, NN, regularized linear. No: trees |
| K-Means k | Elbow, silhouette |
| PCA | Orthogonal max-variance directions; scale first |
| Vanishing gradient | ReLU, BatchNorm, ResNet skip, LSTM |
| Dropout | Randomly zero neurons during training → regularization |
| Attention | softmax(QKᵀ/√d)V |
| RAG | Retrieve relevant docs → feed to LLM |
| A/B sample size | n ≈ 16σ²/δ² per group |
| Peeking | Inflates false positives |
| SRM | Unequal traffic split → assignment bug |
| Simpson's paradox | Aggregated trend reverses in subgroups |
| Stationarity | Constant mean/var; ADF test; differencing |
| RANK vs DENSE_RANK | 1,2,2,4 vs 1,2,2,3 |
| WHERE vs HAVING | Before vs after GROUP BY |
