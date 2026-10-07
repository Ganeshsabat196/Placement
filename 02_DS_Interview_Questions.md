# Data Scientist Interview Questions with Answers (Round 1 — Concepts)

> **How to practice**: read the question, **answer out loud first**, then check the answer. Mark weak ones with ❌ and revisit.
> Difficulty: 🟢 Easy · 🟡 Medium · 🔴 Hard

## Sections
1. [Statistics](#1-statistics)
2. [Probability (with puzzles)](#2-probability)
3. [Hypothesis Testing & A/B Testing](#3-hypothesis-testing--ab-testing)
4. [ML Fundamentals](#4-ml-fundamentals)
5. [Regression](#5-regression)
6. [Classification & Algorithms](#6-classification--algorithms)
7. [Trees & Ensembles](#7-trees--ensembles)
8. [Unsupervised Learning](#8-unsupervised-learning)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Data Preprocessing & Feature Engineering](#10-data-preprocessing--feature-engineering)
11. [Deep Learning](#11-deep-learning)
12. [NLP & GenAI](#12-nlp--genai)
13. [Time Series](#13-time-series)
14. [SQL (with queries)](#14-sql)
15. [Python & Pandas](#15-python--pandas)
16. [MLOps / Deployment](#16-mlops--deployment)
17. [Project & HR-style Technical Questions](#17-project--behavioral-technical-questions)

---

## 1. Statistics

**Q1. 🟢 What is the difference between mean, median and mode? When would you use median over mean?**
Mean = average, median = middle value, mode = most frequent. Use median when data is skewed or has outliers (e.g., house prices, salaries) because the mean gets pulled toward extreme values.

**Q2. 🟢 What is standard deviation vs variance?**
Variance is the average squared deviation from the mean; SD is its square root, so it's in the same unit as the data and easier to interpret.

**Q3. 🟡 Why do we divide by n−1 for sample variance?**
Because we use the sample mean (estimated from the same data), which makes deviations slightly smaller than they'd be from the true mean. Dividing by n−1 (Bessel's correction) makes the estimator unbiased. We lose one degree of freedom.

**Q4. 🟢 What is skewness? Give a real example of right-skewed data.**
Asymmetry of a distribution. Right-skewed: long tail on the right — income, house prices, time on website. Mean > median.

**Q5. 🟡 How do you detect and treat outliers?**
Detect: box plot/IQR rule, z-score > 3, scatter plots, Isolation Forest/DBSCAN. Treat: first check if it's an error (fix/remove) or genuine (keep). Options: cap/winsorize, log transform, use robust models/metrics (median, MAE, tree models), or model separately. Never blindly delete — in fraud detection outliers are the signal.

**Q6. 🟢 Difference between covariance and correlation?**
Covariance shows direction of linear relationship but depends on units. Correlation is standardized covariance (−1 to 1), unit-free, shows both strength and direction.

**Q7. 🟡 Pearson vs Spearman correlation?**
Pearson measures linear relationship, sensitive to outliers. Spearman uses ranks → measures monotonic relationship, robust to outliers and works for ordinal data.

**Q8. 🟢 Correlation vs causation — give an example.**
Ice cream sales and drowning deaths are correlated, but both are caused by hot weather (confounder). To establish causation: randomized experiments, or causal inference methods (DiD, IV, matching).

**Q9. 🟡 What is the Central Limit Theorem and why is it important?**
The distribution of sample means approaches a normal distribution as sample size increases (n≥30 typically), regardless of the population's distribution, with mean μ and SE σ/√n. It allows us to use z/t-tests and build confidence intervals even when the data isn't normal — the foundation of A/B testing.

**Q10. 🟡 What is the Law of Large Numbers? How is it different from CLT?**
LLN: the sample mean converges to the population mean as n grows. CLT: describes the *shape* (normal) of the distribution of the sample mean. LLN is about the value; CLT is about the distribution.

**Q11. 🟢 What is a normal distribution? Properties?**
Bell-shaped, symmetric, mean = median = mode, defined by μ and σ. 68-95-99.7 rule.

**Q12. 🟡 How would you check if data is normally distributed?**
Histogram, Q-Q plot (points on straight line), skewness/kurtosis values, Shapiro-Wilk test (small n), Kolmogorov-Smirnov / Anderson-Darling.

**Q13. 🟡 What are different types of sampling? What is sampling bias?**
Simple random, stratified, cluster, systematic, convenience. Sampling bias: sample doesn't represent population (e.g., online survey excludes people without internet).

**Q14. 🟡 What is selection bias and survivorship bias?**
Selection bias: non-random selection makes sample unrepresentative. Survivorship bias: only looking at "survivors" — e.g., WWII planes: armor was added where returning planes were *not* hit, because planes hit there didn't return.

**Q15. 🟢 What is a confidence interval? Interpret a 95% CI.**
A range likely to contain the true parameter. If we repeated sampling 100 times, about 95 of the computed intervals would contain the true value.

**Q16. 🟡 What is Simpson's paradox?**
A trend that appears in several groups reverses when the groups are combined, due to a confounding variable with unequal group sizes. Example: Treatment A has better success rate for both small and large kidney stones, but B looks better overall because B was given mostly to easy (small stone) cases.

**Q17. 🔴 What is the difference between a standard deviation and standard error?**
SD measures spread of individual data points. SE measures spread of a sample statistic (e.g., mean) across samples: SE = SD/√n. SE decreases with larger n; SD doesn't.

**Q18. 🟡 What is a long-tailed distribution? Why does it matter?**
Many rare events with a few extremely frequent ones (e.g., product sales, word frequencies). Matters for: choosing metrics (median), modeling (log transform), recommendation (long-tail items), and classification (rare classes).

---

## 2. Probability

**Q19. 🟢 Explain Bayes' theorem with an example.**
P(A|B) = P(B|A)P(A)/P(B). Disease test: 1% prevalence, 99% sensitivity, 5% FPR. P(disease|+) = (0.99×0.01)/(0.99×0.01 + 0.05×0.99) ≈ 16.7%. Even with an accurate test, a positive result is mostly false because the disease is rare.

**Q20. 🟢 Two dice are rolled. Probability the sum is 7?**
Favorable: (1,6),(2,5),(3,4),(4,3),(5,2),(6,1) = 6 out of 36 = **1/6**.

**Q21. 🟢 Probability of at least one head in 3 coin flips?**
1 − P(no heads) = 1 − (1/2)³ = **7/8**.

**Q22. 🟡 A family has two children. At least one is a boy. Probability both are boys?**
Sample space: BB, BG, GB (GG excluded). **1/3**.

**Q23. 🟡 Monty Hall problem — should you switch?**
Yes. Initially you pick the car with 1/3. The host, who knows, opens a goat door. Switching wins with probability **2/3**.

**Q24. 🟡 You flip a fair coin 10 times. Probability of exactly 5 heads?**
C(10,5)/2¹⁰ = 252/1024 ≈ **24.6%**.

**Q25. 🟡 Expected number of rolls of a die to get a 6?**
Geometric distribution with p=1/6 → E = 1/p = **6**.

**Q26. 🟡 Birthday problem: How many people for >50% chance two share a birthday?**
**23** people. P(no match) = 365/365 × 364/365 × … ; drops below 0.5 at n=23.

**Q27. 🔴 A bag has 3 red and 2 blue balls. Draw 2 without replacement. P(both red)?**
(3/5)×(2/4) = **3/10**.

**Q28. 🔴 You have a coin; you suspect it's biased. 10 flips gives 8 heads. Is it biased?**
H₀: p=0.5. P(≥8 heads) = (C(10,8)+C(10,9)+C(10,10))/1024 = (45+10+1)/1024 ≈ 0.055 (one-tailed); two-tailed ≈ 0.11. Not significant at α=0.05 → can't conclude bias; need more flips.

**Q29. 🟡 What is the difference between independent and mutually exclusive events?**
Independent: one doesn't affect the other's probability, P(A∩B)=P(A)P(B). Mutually exclusive: can't happen together, P(A∩B)=0. If both have non-zero probability, mutually exclusive events are dependent (if A happens, B certainly doesn't).

**Q30. 🟡 When would you use Poisson vs Binomial distribution?**
Binomial: fixed number of trials n, count successes (e.g., 10 emails, how many opened). Poisson: count of events in a fixed interval with no fixed n (e.g., customers per hour). Poisson approximates Binomial when n is large and p small.

**Q31. 🔴 A stick is broken at two random points. Probability the three pieces form a triangle?**
**1/4**. (Each piece must be less than half the length.)

**Q32. 🔴 Expected number of coin flips to get two heads in a row?**
**6**. Let E = expected flips. Set up states: E = 1 + ½E₁ + ½E, E₁ = 1 + ½·0 + ½E → solving gives E = 6.

---

## 3. Hypothesis Testing & A/B Testing

**Q33. 🟢 What is a p-value? Explain to a non-technical person.**
Technical: probability of seeing results at least this extreme if the null hypothesis were true. Simple: "If the new feature truly had no effect, how surprising would our results be? A small p-value means very surprising, so we believe the feature probably has an effect."

**Q34. 🟢 What are Type I and Type II errors? Which is worse?**
Type I: false positive (convict an innocent person). Type II: false negative (let a guilty person go). Which is worse depends on context — in cancer screening Type II is worse; in spam filtering (blocking important email) Type I is worse.

**Q35. 🟡 What is statistical power? How do you increase it?**
Probability of detecting a true effect (1−β). Increase by: larger sample, larger effect size, higher α, lower variance (CUPED, better metrics), one-tailed test.

**Q36. 🟡 When do you use a t-test vs z-test?**
z-test: population SD known or large n (>30). t-test: SD unknown and small sample; t-distribution has heavier tails. For large n they converge.

**Q37. 🟡 What test would you use to compare conversion rates of two groups?**
Two-proportion z-test (or chi-square test on a 2×2 table — equivalent).

**Q38. 🟡 What is ANOVA? Why not use multiple t-tests?**
ANOVA tests whether means of 3+ groups differ, using ratio of between-group to within-group variance (F-statistic). Multiple t-tests inflate the Type I error rate. After significant ANOVA, use post-hoc tests (Tukey HSD).

**Q39. 🟡 What is a chi-square test used for?**
Test of independence between two categorical variables (e.g., gender vs product preference), or goodness-of-fit (observed vs expected frequencies). Also used for SRM check in A/B testing.

**Q40. 🟡 Walk me through how you would design an A/B test.**
1) Define hypothesis & business goal. 2) Choose primary metric + guardrail metrics. 3) Choose randomization unit (user). 4) Calculate sample size from baseline, MDE, α=0.05, power=0.8. 5) Decide duration (≥1–2 full weeks). 6) Randomly split, run without peeking. 7) Sanity checks (SRM, A/A, invariant metrics). 8) Analyze with appropriate test, CI, segment analysis. 9) Make decision considering practical significance and costs.

**Q41. 🔴 How do you calculate sample size for an A/B test?**
Inputs: baseline rate, minimum detectable effect, α, power. Rule of thumb n ≈ 16·p(1−p)/δ² per group. E.g., baseline 5%, detect 0.5% absolute lift: 16×0.0475/0.000025 ≈ 30,400 per group.

**Q42. 🔴 What is the problem with "peeking" at A/B test results?**
Checking repeatedly and stopping when p<0.05 dramatically increases false positives (could be 20–30% instead of 5%), because random fluctuations will cross the threshold at some point. Fix: pre-define sample size, or use sequential testing (O'Brien-Fleming, alpha spending, always-valid p-values).

**Q43. 🔴 Your A/B test shows a significant lift, but the PM is skeptical. What would you check?**
SRM (traffic split correct?), A/A test results, novelty effect (lift decaying over time?), test duration covering weekly cycle, multiple testing, segment consistency, outliers driving the mean, instrumentation bugs, guardrail metrics, whether the effect is practically significant.

**Q44. 🔴 What are network effects in A/B testing and how do you handle them?**
When treatment users influence control users (e.g., a new messaging feature in a social app; ride-sharing marketplace where treatment drivers take riders from control). Violates independence (SUTVA). Solutions: cluster-based randomization (by friend graph cluster), geo/market-level randomization, switchback experiments (alternate treatment over time).

**Q45. 🟡 What is a novelty effect?**
Users initially engage more with a new feature just because it's new; effect fades. Detect by plotting the treatment effect over time or analyzing new vs returning users. Run tests longer.

**Q46. 🔴 What if you can't run an A/B test? How do you measure impact?**
Quasi-experimental methods: Difference-in-Differences (compare change in treated vs untreated region before/after), Regression Discontinuity, Propensity Score Matching, Synthetic Control, Interrupted Time Series, Instrumental Variables.

**Q47. 🟡 What is the multiple testing problem and how to fix it?**
More tests → higher chance of false positives. With 20 tests at α=0.05, P(≥1 false positive) ≈ 64%. Fix: Bonferroni (α/m), Holm, Benjamini-Hochberg (FDR control).

**Q48. 🟡 Statistical significance vs practical significance?**
Statistically significant = unlikely due to chance. Practically significant = large enough to matter for the business. With millions of users, a 0.01% lift can be significant but not worth the engineering/maintenance cost.

**Q49. 🔴 What is CUPED?**
Controlled-experiment Using Pre-Experiment Data. Adjusts the metric using a pre-experiment covariate (e.g., user's past spend): Y_adj = Y − θ(X − mean(X)). Reduces variance → smaller sample sizes / faster tests.

**Q50. 🟡 What are guardrail metrics?**
Metrics that must not degrade even if the primary metric improves — e.g., page load time, crash rate, unsubscribe rate, revenue. They protect against harmful launches.

---

## 4. ML Fundamentals

**Q51. 🟢 What is the difference between supervised, unsupervised and reinforcement learning?**
Supervised: learn mapping from labeled inputs to outputs (spam detection). Unsupervised: find structure in unlabeled data (customer segmentation). RL: agent learns actions through rewards (game playing, robotics).

**Q52. 🟢 Regression vs classification?**
Regression predicts continuous values (price); classification predicts categories (spam/not spam).

**Q53. 🟡 Explain the bias-variance trade-off.**
Bias: error from overly simple assumptions (underfitting). Variance: error from sensitivity to training data fluctuations (overfitting). As complexity increases, bias falls and variance rises. Goal: sweet spot minimizing total error. Example: linear model on curved data = high bias; deep unpruned tree = high variance.

**Q54. 🟢 What is overfitting and how do you prevent it?**
Model learns noise in training data → great train performance, poor test performance. Prevent: more data, cross-validation, regularization (L1/L2), simpler model, pruning, dropout, early stopping, feature selection, ensembling (bagging), data augmentation.

**Q55. 🟡 How do you know if your model is overfitting or underfitting?**
Compare training vs validation error / learning curves. Overfitting: low train error, high val error (big gap). Underfitting: both high.

**Q56. 🟡 What is regularization? Why does it work?**
Adding a penalty for model complexity (large weights) to the loss function. It constrains the model, preventing it from fitting noise, trading a bit of bias for a large reduction in variance.

**Q57. 🟡 Explain gradient descent. What are its variants?**
Iterative optimization: move parameters in the direction of negative gradient of the loss, step size = learning rate. Variants: batch (full data), stochastic (1 sample), mini-batch. Advanced: momentum, RMSprop, Adam.

**Q58. 🟡 What happens if the learning rate is too high or too low?**
Too high: overshoots minimum, oscillates or diverges. Too low: very slow convergence, can get stuck in plateaus. Use LR schedules or adaptive optimizers.

**Q59. 🟡 What is the curse of dimensionality?**
As the number of features grows, data becomes sparse, distances between points become similar (distance-based methods like KNN fail), and you need exponentially more data. Fix: feature selection, PCA, regularization.

**Q60. 🟡 Parametric vs non-parametric models?**
Parametric: fixed number of parameters, assumes functional form (linear regression). Non-parametric: no fixed form, complexity grows with data (KNN, decision trees, kernel SVM).

**Q61. 🔴 Generative vs discriminative models?**
Discriminative learn P(y|x) — decision boundary (logistic regression, SVM). Generative learn P(x,y) or P(x|y) — how data is generated (Naive Bayes, GMM, HMM, GANs). Generative can generate new samples and work with less data; discriminative are usually more accurate for classification.

**Q62. 🟡 What is the difference between a parameter and a hyperparameter?**
Parameters are learned from data (weights, coefficients). Hyperparameters are set before training (learning rate, k in KNN, max_depth) and tuned via validation.

**Q63. 🔴 What is MLE (Maximum Likelihood Estimation)?**
Choose parameters that maximize the likelihood of the observed data. Linear regression with Gaussian noise → MLE equals minimizing MSE. Logistic regression → MLE equals minimizing log loss.

**Q64. 🟡 How do you choose which algorithm to use?**
Depends on: problem type, data size, number of features, interpretability needs, latency, linearity, data types. Start with a simple baseline (logistic/linear regression), then try tree ensembles (RF, XGBoost) for tabular data, deep learning for images/text/audio. Validate with CV and pick based on business metric + constraints.

---

## 5. Regression

**Q65. 🟢 What are the assumptions of linear regression?**
Linearity, independence of errors, homoscedasticity (constant error variance), normality of residuals, no multicollinearity.

**Q66. 🟡 What happens if these assumptions are violated?**
Non-linearity → biased predictions (add polynomial terms/transform). Heteroscedasticity → unreliable standard errors/p-values (use log transform, weighted least squares, robust SEs). Autocorrelation → underestimated SEs (time-series models). Multicollinearity → unstable coefficients (drop features, Ridge, PCA). Non-normal residuals → affects inference in small samples.

**Q67. 🟡 What is R²? Can it be negative?**
Proportion of variance in y explained by the model. On training data with intercept, it's between 0 and 1. On test data it can be negative — model is worse than predicting the mean.

**Q68. 🟡 R² vs Adjusted R²?**
R² never decreases when adding features (even useless ones). Adjusted R² penalizes the number of predictors and only increases if the new feature improves the model more than expected by chance.

**Q69. 🟡 What is multicollinearity? How do you detect and handle it?**
High correlation between independent variables → unstable, high-variance coefficients, hard to interpret. Detect: correlation matrix, VIF > 5–10. Handle: remove one of the correlated features, combine them, PCA, Ridge regression.

**Q70. 🟡 L1 vs L2 regularization? Why does L1 produce sparse solutions?**
L1 (Lasso) adds λΣ|β|; L2 (Ridge) adds λΣβ². L1 can shrink coefficients exactly to zero → feature selection. Geometrically, the L1 constraint is a diamond with corners on axes; the elliptical loss contours usually touch the diamond at a corner where some coefficients = 0. L2's circle has no corners, so coefficients become small but not zero.

**Q71. 🟡 How do you interpret coefficients in linear regression? And with log-transformed target?**
β₁: one unit increase in x₁ changes y by β₁, others constant. If y is log-transformed: one-unit increase in x changes y by about (e^β − 1)×100%.

**Q72. 🟡 MAE vs MSE vs RMSE — which to use?**
MAE: robust to outliers, all errors weighted equally. MSE/RMSE: penalize large errors more — use when large errors are particularly bad. RMSE is in the target's units.

**Q73. 🔴 Why can't we use linear regression for classification?**
Outputs aren't bounded to [0,1], sensitive to outliers shifting the decision boundary, errors aren't normally distributed, and MSE is not appropriate for probabilities. Use logistic regression.

**Q74. 🟡 What are residuals? What does a residual plot tell you?**
Residual = actual − predicted. A good residual plot shows random scatter around 0. Patterns (curve) → non-linearity; funnel shape → heteroscedasticity.

---

## 6. Classification & Algorithms

**Q75. 🟢 How does logistic regression work?**
Computes a linear combination of features, passes it through a sigmoid to get a probability between 0 and 1, and classifies with a threshold. It models the log-odds as linear in features. Trained by maximizing likelihood (minimizing log loss).

**Q76. 🟡 Why is logistic regression called "regression"?**
It regresses the log-odds of the outcome on the features (a linear regression on the logit scale); classification only happens after applying a threshold.

**Q77. 🟡 How to interpret logistic regression coefficients?**
β = change in log-odds per unit increase of x. e^β = odds ratio; e.g., β=0.7 → e^0.7≈2 → odds double.

**Q78. 🟡 Why not use MSE as loss for logistic regression?**
With sigmoid, MSE makes the loss non-convex (multiple local minima) and gives weak gradients. Log loss is convex and penalizes confident wrong predictions heavily.

**Q79. 🟡 How does KNN work? How do you choose k?**
For a new point, find k closest training points (by distance) and take majority vote (classification) or average (regression). Choose k via cross-validation; small k = overfitting/noise-sensitive, large k = oversmoothing. Must scale features.

**Q80. 🟡 What is "naive" in Naive Bayes? When does it work well?**
It assumes features are conditionally independent given the class (rarely true). Still works well for text classification/spam, high-dimensional data, small datasets, and is very fast.

**Q81. 🟡 Explain SVM. What is the kernel trick?**
SVM finds the hyperplane maximizing the margin between classes; only support vectors (points closest to boundary) define it. Kernel trick computes dot products in a higher-dimensional space without explicitly transforming data, allowing non-linear boundaries (RBF, polynomial).

**Q82. 🟡 What does the C parameter in SVM do?**
Controls trade-off between maximizing margin and minimizing misclassification. High C = penalize errors heavily → narrow margin, risk of overfitting. Low C = wider margin, more tolerance → may underfit.

**Q83. 🔴 Logistic regression vs SVM vs Decision Tree — when to use which?**
Logistic: need probabilities, interpretability, linearly separable, baseline. SVM: high-dimensional, clear margin, small/medium datasets. Trees: non-linear relationships, mixed data types, interpretability via rules; ensembles of trees for best tabular performance.

**Q84. 🔴 How would you handle multi-class classification?**
Natively multi-class algorithms (trees, Naive Bayes, softmax regression, NN) or decompose: One-vs-Rest (k classifiers) or One-vs-One (k(k−1)/2 classifiers). Evaluate with macro/micro/weighted F1 and a confusion matrix.

---

## 7. Trees & Ensembles

**Q85. 🟢 How does a decision tree decide where to split?**
Tries all features and thresholds, picks the split that maximizes impurity reduction — Gini impurity or entropy/information gain for classification, variance reduction (MSE) for regression.

**Q86. 🟡 Gini vs Entropy?**
Both measure impurity. Gini = 1−Σp², entropy = −Σp log p. Gini is slightly faster (no log); results are usually very similar.

**Q87. 🟡 Why do decision trees overfit? How do you prevent it?**
They can keep splitting until every leaf is pure, memorizing noise. Prevent: max_depth, min_samples_leaf, min_samples_split, max_leaf_nodes, pruning (cost complexity), or use ensembles.

**Q88. 🟡 Bagging vs Boosting?**
Bagging: train many models independently on bootstrap samples in parallel, average them → reduces variance (Random Forest). Boosting: train models sequentially, each focusing on previous errors → reduces bias (XGBoost). Boosting is more prone to overfitting noise and needs careful tuning.

**Q89. 🟡 How does Random Forest work? Why is it better than a single tree?**
Builds many decision trees on bootstrap samples and at each split considers a random subset of features; aggregates by vote/average. Randomness decorrelates trees, so averaging reduces variance significantly while keeping bias low.

**Q90. 🟡 What is out-of-bag (OOB) error?**
Each tree's bootstrap sample excludes ~37% of rows. Predicting those rows with trees that didn't see them gives a free validation estimate.

**Q91. 🔴 How does Gradient Boosting work?**
Start with a simple prediction (mean). Compute residuals (negative gradients of loss). Fit a small tree to the residuals. Add it to the model multiplied by learning rate. Repeat. Final prediction = sum of all trees.

**Q92. 🔴 What makes XGBoost better than standard GBM?**
Regularization (L1/L2 on leaf weights), second-order gradient (Newton) optimization, built-in missing value handling, parallelized split finding, tree pruning with max_depth & gamma, column/row subsampling, efficient cache usage.

**Q93. 🟡 XGBoost vs LightGBM vs CatBoost?**
XGBoost: level-wise growth, very robust. LightGBM: leaf-wise growth + histogram binning → much faster on large data, risk of overfitting small data. CatBoost: best native categorical handling, ordered boosting reduces target leakage, good defaults.

**Q94. 🟡 Random Forest vs XGBoost — which would you choose?**
RF: fewer hyperparameters, robust, parallel, less overfitting — good quick baseline. XGBoost: usually higher accuracy on tabular data when tuned, handles imbalance (scale_pos_weight), but sensitive to hyperparameters.

**Q95. 🔴 How is feature importance calculated in tree models? What are its pitfalls?**
Impurity-based: total reduction in impurity from splits on that feature. Pitfalls: biased toward high-cardinality/continuous features, misleading with correlated features. Better: permutation importance or SHAP values.

**Q96. 🟡 What is stacking?**
Train several diverse base models, then train a meta-model on their (out-of-fold) predictions to learn the best combination.

---

## 8. Unsupervised Learning

**Q97. 🟢 Explain K-Means clustering.**
Initialize k centroids, assign each point to the nearest centroid, update centroids to the mean of assigned points, repeat until assignments stop changing. Minimizes within-cluster sum of squares.

**Q98. 🟡 How do you choose k in K-Means?**
Elbow method (plot inertia vs k, look for the elbow), silhouette score (maximize), gap statistic, and business interpretability (e.g., marketing can only handle 4 segments).

**Q99. 🟡 Limitations of K-Means?**
Need to specify k, assumes spherical equal-size clusters, sensitive to initialization (use k-means++), outliers, and feature scale; only works with numeric data (use K-Modes/K-Prototypes for categorical).

**Q100. 🟡 K-Means vs DBSCAN vs Hierarchical?**
K-Means: fast, needs k, spherical clusters. DBSCAN: density-based, arbitrary shapes, detects noise/outliers, no k needed, struggles with varying density. Hierarchical: produces dendrogram, no k upfront, O(n²) memory — not for large data.

**Q101. 🟡 What is PCA? How does it work?**
Dimensionality reduction that projects data onto orthogonal directions (principal components) capturing maximum variance. Steps: standardize, compute covariance matrix, eigen-decomposition, sort by eigenvalues, keep top k components.

**Q102. 🟡 Why scale data before PCA?**
PCA maximizes variance; features with larger scale (e.g., salary vs age) would dominate the components.

**Q103. 🟡 How do you decide the number of principal components?**
Cumulative explained variance (e.g., keep 90–95%), scree plot elbow, or downstream model performance.

**Q104. 🔴 PCA vs t-SNE vs LDA?**
PCA: linear, unsupervised, preserves global variance, fast. t-SNE/UMAP: non-linear, preserves local neighborhoods, mainly for visualization. LDA: supervised, maximizes class separability.

**Q105. 🟡 How would you evaluate a clustering model without labels?**
Silhouette score, Davies-Bouldin index, Calinski-Harabasz, cluster stability, and most importantly business interpretability — profile clusters and check if they're actionable.

**Q106. 🟡 How would you do anomaly detection?**
Statistical (z-score, IQR), Isolation Forest (anomalies isolated in fewer splits), Local Outlier Factor, One-class SVM, autoencoders (high reconstruction error), or supervised if labels exist.

---

## 9. Evaluation Metrics

**Q107. 🟢 What is a confusion matrix?**
Table of TP, FP, FN, TN comparing predicted vs actual classes. Basis for precision, recall, accuracy, F1.

**Q108. 🟢 Precision vs Recall — give examples where each matters more.**
Precision = TP/(TP+FP); matters when false positives are costly — spam filter (don't block important mail), recommending stocks. Recall = TP/(TP+FN); matters when false negatives are costly — cancer detection, fraud detection, airport security.

**Q109. 🟡 Why is accuracy a bad metric for imbalanced data?**
With 99% negatives, predicting "negative" always gives 99% accuracy while catching zero positives. Use precision, recall, F1, PR-AUC.

**Q110. 🟡 What is the F1-score? Why harmonic mean?**
F1 = 2PR/(P+R). Harmonic mean punishes extreme values — if either precision or recall is low, F1 is low.

**Q111. 🟡 Explain ROC curve and AUC.**
ROC plots TPR vs FPR at all thresholds. AUC = area under it = probability the model ranks a random positive higher than a random negative. 0.5 = random guessing, 1 = perfect. Threshold-independent.

**Q112. 🔴 When would you prefer PR-AUC over ROC-AUC?**
For highly imbalanced data. ROC uses FPR = FP/(FP+TN), and with huge TN, FPR stays low even with many FPs → ROC looks optimistic. PR curve focuses on the positive class.

**Q113. 🟡 How do you choose a classification threshold?**
Based on business costs: assign costs to FP and FN, pick threshold that minimizes total cost or meets a constraint (e.g., recall ≥ 90%). Use PR curve, ROC (Youden's J), or capacity constraints (only 100 cases can be manually reviewed per day).

**Q114. 🔴 What is model calibration?**
Whether predicted probabilities match observed frequencies (of all cases predicted 0.7, ~70% should be positive). Check with reliability diagram / Brier score. Fix with Platt scaling or isotonic regression. Important when probabilities drive decisions (expected revenue, risk pricing).

**Q115. 🟡 What is log loss?**
−[y log p + (1−y) log(1−p)] averaged. Heavily penalizes confident wrong predictions. Measures probability quality.

**Q116. 🟡 Macro vs micro vs weighted averaging?**
Macro: average metric per class equally (treats rare classes as important). Micro: aggregate TP/FP/FN globally (dominated by frequent classes). Weighted: per-class average weighted by support.

**Q117. 🟡 What metrics would you use for a recommendation/ranking system?**
Offline: Precision@K, Recall@K, MAP, NDCG, MRR, hit rate, coverage, diversity. Online: CTR, conversion, watch time, retention via A/B test.

---

## 10. Data Preprocessing & Feature Engineering

**Q118. 🟢 How do you handle missing data?**
First understand why it's missing (MCAR/MAR/MNAR). Options: drop (if few rows/useless column), impute with mean/median/mode, KNN or iterative imputation, forward-fill for time series, add a missing indicator, or use models that handle NaN (XGBoost). Imputation statistics must be computed on training data only.

**Q119. 🟡 What are MCAR, MAR and MNAR?**
MCAR: missingness unrelated to any data (random sensor failure). MAR: related to other observed variables (older people skip an online question). MNAR: related to the missing value itself (high earners don't report income) — hardest to handle.

**Q120. 🟡 One-hot encoding vs label encoding?**
Label encoding assigns integers — fine for ordinal data and tree models, but implies false order for nominal data in linear models. One-hot creates binary columns — for nominal data, but explodes dimensions with high cardinality.

**Q121. 🔴 How do you handle high-cardinality categorical features (e.g., 10,000 zip codes)?**
Target/mean encoding (with K-fold and smoothing to avoid leakage), frequency encoding, grouping rare categories into "Other", hashing trick, embeddings (NN), or CatBoost's native handling. Also map to higher-level groups (zip → city/region).

**Q122. 🟡 Standardization vs normalization? Which models need scaling?**
Standardization: mean 0, SD 1 (robust-ish, for normally distributed features). Normalization: scales to [0,1] (sensitive to outliers). Need scaling: KNN, SVM, K-Means, PCA, linear/logistic with regularization, neural networks. Don't need: tree-based models.

**Q123. 🟡 How do you handle imbalanced datasets?**
Proper metrics (PR-AUC, recall, F1), class weights, resampling (undersample majority, oversample minority, SMOTE — only on training data), threshold tuning, ensemble methods (balanced RF, EasyEnsemble), anomaly detection framing, collect more minority data.

**Q124. 🔴 How does SMOTE work? What are its drawbacks?**
For each minority sample, pick one of its k nearest minority neighbors and create a synthetic sample on the line between them. Drawbacks: can create noisy samples in overlapping regions, doesn't work well with high dimensions or categorical data, must be applied after the train/test split (inside CV folds) to avoid leakage.

**Q125. 🔴 What is data leakage? Give examples.**
When training uses information unavailable at prediction time → unrealistically good validation results that fail in production. Examples: fitting scaler/imputer on full data before splitting; including a feature like "account_closed_date" when predicting churn; random split for time-series; same patient in train and test; target encoding without CV.

**Q126. 🟡 What feature selection methods do you know?**
Filter (correlation, chi-square, mutual information, variance threshold), wrapper (RFE, forward/backward selection), embedded (Lasso, tree importance), plus SHAP / permutation importance.

**Q127. 🟡 Why is feature engineering important? Give examples.**
Good features often matter more than model choice. Examples: from timestamp → hour, weekday, is_weekend; from transactions → RFM (recency, frequency, monetary); ratios (debt/income); aggregates (avg spend last 30 days); text length; distance from home location (fraud).

**Q128. 🟡 How do you deal with skewed features?**
Log, square root, Box-Cox, Yeo-Johnson transforms; binning; or use tree models that are insensitive to monotonic transforms.

---

## 11. Deep Learning

**Q129. 🟢 What is a neural network? How does it learn?**
Layers of neurons computing weighted sums plus non-linear activations. Learns by forward pass (prediction), computing loss, backpropagation (gradients via chain rule), and updating weights with gradient descent.

**Q130. 🟡 Explain backpropagation.**
Algorithm to compute the gradient of the loss with respect to every weight by applying the chain rule backward from output to input layer, reusing intermediate gradients for efficiency.

**Q131. 🟡 Why do we need activation functions? Why is ReLU popular?**
Without non-linearity, a deep network is equivalent to a single linear layer. ReLU (max(0,x)) is cheap, doesn't saturate for positive inputs (reduces vanishing gradients), and gives sparse activations. Drawback: dying ReLU → Leaky ReLU.

**Q132. 🟡 What is the vanishing gradient problem?**
In deep networks with sigmoid/tanh, gradients get multiplied by small numbers through many layers, becoming tiny → early layers don't learn. Solutions: ReLU, proper initialization (He/Xavier), batch normalization, residual connections, LSTM/GRU for sequences.

**Q133. 🟡 What is dropout?**
During training, randomly set a fraction of neurons to zero each step. Prevents co-adaptation, acts like training an ensemble of sub-networks → regularization. Disabled at inference (outputs scaled accordingly).

**Q134. 🟡 What is batch normalization?**
Normalizes layer inputs to zero mean, unit variance per mini-batch, then applies learnable scale & shift. Stabilizes and speeds up training, allows higher learning rates, slight regularization.

**Q135. 🟡 SGD vs Adam?**
SGD uses one global learning rate; with momentum it often generalizes well but needs tuning. Adam uses adaptive per-parameter learning rates with momentum (first and second moments) → converges faster, good default.

**Q136. 🟡 What is a CNN? Why use it for images?**
Uses convolutional filters that slide across the image to detect local patterns, with parameter sharing and pooling. Far fewer parameters than fully connected networks, captures spatial hierarchy (edges→textures→objects), translation invariant.

**Q137. 🟡 What is transfer learning?**
Reusing a model pretrained on a large dataset (ImageNet, BERT) for a new task — freeze early layers (general features), fine-tune later layers. Works well with little data and saves compute.

**Q138. 🟡 RNN vs LSTM vs GRU?**
RNN processes sequences with a hidden state but suffers vanishing gradients for long dependencies. LSTM adds a cell state and three gates (forget, input, output) to retain long-term information. GRU simplifies to two gates — faster, similar performance.

**Q139. 🔴 Explain the Transformer and self-attention.**
Self-attention lets each token compute a weighted sum of all other tokens' values, with weights from the similarity of its query to their keys: softmax(QKᵀ/√d)V. Multi-head attention captures different relationships. Transformers add positional encodings, feed-forward layers, residual connections and layer norm. Advantages over RNN: parallel processing and better long-range dependencies.

**Q140. 🟡 What is an epoch, batch, and iteration?**
Epoch: one full pass through training data. Batch: subset of samples processed before one weight update. Iteration: one update step. Iterations per epoch = n / batch size.

**Q141. 🟡 How would you deal with overfitting in a neural network?**
More data / augmentation, dropout, L2 weight decay, early stopping, smaller network, batch norm, transfer learning.

**Q142. 🟡 What is an autoencoder?**
Network trained to reconstruct its input through a narrow bottleneck: learns compressed representations. Uses: dimensionality reduction, denoising, anomaly detection (high reconstruction error).

---

## 12. NLP & GenAI

**Q143. 🟢 Stemming vs lemmatization?**
Stemming chops word endings by rules (studies → studi), fast but crude. Lemmatization returns the dictionary base form using vocabulary/POS (studies → study, better → good), more accurate but slower.

**Q144. 🟡 What is TF-IDF?**
Term Frequency × Inverse Document Frequency. Weights a word highly if it's frequent in a document but rare across the corpus, so common words like "the" get low weight.

**Q145. 🟡 Bag-of-Words/TF-IDF vs Word Embeddings?**
BoW/TF-IDF: sparse, high-dimensional, no semantic meaning or word order. Embeddings (Word2Vec, GloVe): dense, low-dimensional, similar words have similar vectors. Contextual embeddings (BERT) give different vectors depending on context ("bank" of river vs money).

**Q146. 🟡 How does Word2Vec work?**
Shallow neural network trained to predict context words from a word (Skip-gram) or a word from its context (CBOW). The learned hidden weights become word embeddings.

**Q147. 🟡 BERT vs GPT?**
BERT: encoder-only, bidirectional, trained with masked language modeling — best for understanding (classification, NER, QA). GPT: decoder-only, left-to-right autoregressive, trained on next-token prediction — best for generation.

**Q148. 🟡 What is RAG?**
Retrieval-Augmented Generation: split documents into chunks, embed them, store in a vector database; at query time, retrieve the most similar chunks and pass them to the LLM in the prompt. Grounds answers in up-to-date/private data, reduces hallucinations, no retraining needed.

**Q149. 🟡 Fine-tuning vs RAG vs prompt engineering?**
Prompt engineering: cheapest, no training, first thing to try. RAG: when the model needs external/changing knowledge. Fine-tuning: when you need to change behavior/style/format or domain-specific skills; LoRA makes it cheaper.

**Q150. 🟡 What is hallucination in LLMs and how to reduce it?**
Model generates plausible but false information. Reduce with RAG/grounding, lower temperature, asking it to cite sources, guardrails/verification steps, fine-tuning, and evaluation.

**Q151. 🟡 What does temperature do in LLMs?**
Scales logits before softmax. Low temperature (→0) = deterministic, picks most likely tokens. High temperature = more random/creative outputs.

**Q152. 🔴 How would you evaluate an LLM-based application?**
Task-specific metrics (accuracy on labeled set, ROUGE/BLEU for summarization/translation), retrieval metrics for RAG (recall@k, context relevance), faithfulness/groundedness, human evaluation, LLM-as-a-judge with rubrics, latency and cost, and online metrics (user satisfaction, resolution rate).

---

## 13. Time Series

**Q153. 🟢 What are the components of a time series?**
Trend, seasonality, cyclic patterns, and random noise.

**Q154. 🟡 What is stationarity and why does it matter?**
Statistical properties (mean, variance, autocorrelation) constant over time. Models like ARIMA assume it. Test with ADF/KPSS; make stationary by differencing, log transform, or removing trend/seasonality.

**Q155. 🟡 Explain ARIMA(p,d,q).**
AR(p): regression on p lagged values. I(d): number of differences to make series stationary. MA(q): regression on q lagged forecast errors. Choose p from PACF, q from ACF. SARIMA adds seasonal terms.

**Q156. 🟡 Why can't you use random K-fold cross-validation for time series?**
It leaks future information into training. Use time-based split / walk-forward (expanding or sliding window) validation.

**Q157. 🟡 How would you use ML (e.g., XGBoost) for forecasting?**
Create lag features (sales t−1, t−7), rolling statistics (7-day mean), date features (day of week, month, holiday), external features (price, promotions); train with time-based validation; forecast recursively or directly per horizon.

---

## 14. SQL

**Q158. 🟢 Difference between WHERE and HAVING?**
WHERE filters rows before grouping; HAVING filters groups after aggregation (can use aggregates like COUNT).

**Q159. 🟢 Types of joins?**
INNER (matching rows), LEFT (all left + matches, NULL otherwise), RIGHT, FULL OUTER (all from both), CROSS (Cartesian product), SELF (table joined with itself).

**Q160. 🟡 RANK vs DENSE_RANK vs ROW_NUMBER?**
For values 100,90,90,80: ROW_NUMBER → 1,2,3,4; RANK → 1,2,2,4; DENSE_RANK → 1,2,2,3.

**Q161. 🟡 UNION vs UNION ALL?**
UNION removes duplicates (slower), UNION ALL keeps all rows.

**Q162. 🟡 Find the second highest salary.**
```sql
SELECT MAX(salary) FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);
-- or
SELECT DISTINCT salary FROM employees ORDER BY salary DESC LIMIT 1 OFFSET 1;
```

**Q163. 🟡 Top 3 highest-paid employees in each department.**
```sql
WITH ranked AS (
  SELECT name, dept, salary,
         DENSE_RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS rnk
  FROM employees)
SELECT * FROM ranked WHERE rnk <= 3;
```

**Q164. 🟡 Find customers who never placed an order.**
```sql
SELECT c.customer_id, c.name
FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;
```

**Q165. 🟡 Find duplicate emails.**
```sql
SELECT email, COUNT(*) AS cnt FROM users GROUP BY email HAVING COUNT(*) > 1;
```

**Q166. 🟡 Calculate running total of sales per user.**
```sql
SELECT user_id, order_date, amount,
       SUM(amount) OVER (PARTITION BY user_id ORDER BY order_date) AS running_total
FROM orders;
```

**Q167. 🟡 7-day rolling average of daily revenue.**
```sql
SELECT day, revenue,
       AVG(revenue) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7d
FROM daily_revenue;
```

**Q168. 🔴 Month-over-month revenue growth percentage.**
```sql
WITH monthly AS (
  SELECT DATE_TRUNC('month', order_date) AS month, SUM(amount) AS revenue
  FROM orders GROUP BY 1)
SELECT month, revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
             / LAG(revenue) OVER (ORDER BY month), 2) AS mom_growth_pct
FROM monthly;
```

**Q169. 🔴 Users who logged in on 3 consecutive days.**
```sql
WITH d AS (
  SELECT DISTINCT user_id, login_date FROM logins),
g AS (
  SELECT user_id, login_date,
         login_date - (ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date)) * INTERVAL '1 day' AS grp
  FROM d)
SELECT DISTINCT user_id FROM g GROUP BY user_id, grp HAVING COUNT(*) >= 3;
```
(Trick: date minus row number is constant for consecutive dates — "gaps and islands".)

**Q170. 🔴 Calculate Day-7 retention by signup cohort.**
```sql
SELECT s.signup_date,
       COUNT(DISTINCT s.user_id) AS signups,
       COUNT(DISTINCT a.user_id) AS retained_d7,
       ROUND(100.0 * COUNT(DISTINCT a.user_id) / COUNT(DISTINCT s.user_id), 2) AS d7_retention
FROM signups s
LEFT JOIN activity a
  ON s.user_id = a.user_id AND a.activity_date = s.signup_date + INTERVAL '7 day'
GROUP BY s.signup_date;
```

**Q171. 🟡 Click-through rate per ad using conditional aggregation.**
```sql
SELECT ad_id,
       ROUND(100.0 * SUM(CASE WHEN action='click' THEN 1 ELSE 0 END)
             / NULLIF(SUM(CASE WHEN action='view' THEN 1 ELSE 0 END), 0), 2) AS ctr
FROM ad_events GROUP BY ad_id;
```

**Q172. 🟡 COUNT(*) vs COUNT(col) vs COUNT(DISTINCT col)?**
COUNT(*) counts all rows; COUNT(col) ignores NULLs; COUNT(DISTINCT col) counts unique non-null values.

**Q173. 🟡 What is a CTE? Subquery vs CTE?**
Common Table Expression (WITH clause) — named temporary result set. More readable, reusable within the query, and supports recursion. Subqueries are inline; performance is usually similar.

**Q174. 🟡 What is SQL query execution order?**
FROM/JOIN → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT.

---

## 15. Python & Pandas

**Q175. 🟢 List vs tuple vs set vs dictionary?**
List: ordered, mutable, allows duplicates. Tuple: ordered, immutable (can be dict key). Set: unordered, unique items, fast membership. Dict: key-value pairs, O(1) lookup.

**Q176. 🟡 loc vs iloc?**
loc: label-based (inclusive end). iloc: integer position-based (exclusive end).

**Q177. 🟡 merge vs join vs concat in pandas?**
merge: SQL-style join on columns. join: joins on index. concat: stacks DataFrames along rows/columns.

**Q178. 🟡 apply vs map vs vectorization?**
map: element-wise on Series. apply: function along rows/columns of DataFrame or Series. Vectorized NumPy/pandas operations are much faster than both — avoid loops/apply when possible.

**Q179. 🟡 How do you handle a dataset too large to fit in memory?**
Read in chunks (`chunksize`), select needed columns, downcast dtypes (float64→float32, category type), use Parquet, Dask/Polars/PySpark, or push computation to SQL/database.

**Q180. 🟡 Shallow copy vs deep copy?**
Shallow copy creates a new outer object but references the same nested objects; deep copy recursively copies everything. In pandas, `df2 = df` is not a copy; use `df.copy()`.

**Q181. 🟡 What is a generator? Why use it?**
Function using `yield` that produces items lazily one at a time — memory efficient for large data streams.

**Q182. 🟡 Write code: find top 3 products by revenue per category in pandas.**
```python
df['revenue'] = df['price'] * df['qty']
agg = df.groupby(['category', 'product'], as_index=False)['revenue'].sum()
top3 = (agg.sort_values('revenue', ascending=False)
           .groupby('category').head(3))
```

**Q183. 🟡 Write a function to compute precision and recall from scratch.**
```python
def precision_recall(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return precision, recall
```

**Q184. 🔴 Implement K-Means from scratch (NumPy).**
```python
import numpy as np
def kmeans(X, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)
        labels = dists.argmin(axis=1)
        new = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                        for j in range(k)])
        if np.allclose(new, centroids):
            break
        centroids = new
    return labels, centroids
```

**Q185. 🔴 Implement linear regression with gradient descent.**
```python
def linear_regression_gd(X, y, lr=0.01, epochs=1000):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        y_hat = X @ w + b
        err = y_hat - y
        w -= lr * (2 / n) * (X.T @ err)
        b -= lr * (2 / n) * err.sum()
    return w, b
```

---

## 16. MLOps / Deployment

**Q186. 🟡 How would you deploy a machine learning model?**
Serialize model (pickle/joblib/ONNX) → wrap in API (FastAPI/Flask) → containerize (Docker) → deploy on cloud/Kubernetes/serverless → add logging & monitoring → CI/CD for retraining. Or batch scoring via scheduled jobs (Airflow) if real-time isn't needed.

**Q187. 🟡 What is data drift vs concept drift?**
Data drift: input feature distribution changes (new user demographics). Concept drift: relationship between features and target changes (fraudsters change tactics). Detect via PSI, KS tests, monitoring performance; fix by retraining.

**Q188. 🟡 How do you monitor a model in production?**
Input data quality & drift, prediction distribution, performance metrics when labels arrive, business KPIs, latency, errors, and alerting thresholds; scheduled or triggered retraining.

**Q189. 🟡 What is training-serving skew?**
Features computed differently in training vs production (different code/data), causing performance drop. Avoid with shared feature pipelines / feature store and logging production features.

**Q190. 🟡 Batch vs real-time inference?**
Batch: predictions precomputed periodically (daily churn scores) — cheaper, simpler. Real-time: on request with low latency (fraud check at payment) — needs fast features and model serving infra.

---

## 17. Project & Behavioral Technical Questions

**Q191. Walk me through your best data science project end-to-end.**
Structure: Problem & business impact → Data (source, size) → Cleaning/EDA findings → Feature engineering → Models tried (baseline first) → Evaluation metric and why → Results with numbers → Deployment/next steps → What you learned.

**Q192. What was the biggest challenge in your project and how did you overcome it?**
Pick a real technical challenge (imbalanced data, leakage found, messy data, overfitting) and explain the diagnosis + solution + result.

**Q193. Why did you choose that algorithm/metric?**
Justify with data characteristics, business cost of errors, interpretability, and comparison with alternatives you tried.

**Q194. How would you explain your model to a non-technical stakeholder?**
Avoid jargon, focus on what decisions it supports, show top drivers (feature importance/SHAP) with intuitive examples, quantify business impact, and be clear about limitations.

**Q195. Your model performs well offline but poorly in production. Why?**
Data leakage in training, training-serving skew, data/concept drift, different population (sampling bias), wrong validation split (random vs time), feedback loops, bugs in feature pipeline.

**Q196. Your model has 99% accuracy. Should you be happy?**
Not necessarily — check class imbalance (baseline accuracy), leakage, validation strategy, and business-relevant metrics (precision/recall).

**Q197. How do you approach a new dataset?**
Understand business context → check shape, types, missing values, duplicates → univariate distributions → bivariate relations with target → outliers → data quality issues → form hypotheses → baseline model.

**Q198. How do you stay updated in data science?**
Mention specific blogs, papers, courses, Kaggle, newsletters, recent tools you tried (e.g., LLM/RAG projects).

**Q199. Tell me about a time your analysis changed a decision.**
Use STAR (Situation, Task, Action, Result) with numbers.

**Q200. What would you do in your first 90 days as a data scientist here?**
Learn business & data, meet stakeholders, find quick wins, build trust with reliable analyses, then identify high-impact longer-term projects.

---

### Rapid-fire round (answer in one line)
1. Can a model have high bias and high variance? → Yes (e.g., wrong model + small noisy data).
2. Does feature scaling affect Random Forest? → No.
3. Is logistic regression a linear classifier? → Yes, linear decision boundary.
4. Default k-fold value? → 5 or 10.
5. Does PCA require labels? → No.
6. Is K-Means deterministic? → No, depends on initialization.
7. What does a VIF of 1 mean? → No multicollinearity.
8. What kind of loss for multi-class NN? → Categorical cross-entropy with softmax.
9. Which is faster to train: RF or GBM? → RF (parallelizable).
10. Can AUC be below 0.5? → Yes — model is worse than random (invert predictions).
11. p-value = 0.03, α = 0.05? → Reject H₀.
12. Is high recall always good? → No, may come with many false positives.
13. What does `dropna()` do by default? → Drops rows with any NaN.
14. What is an outlier-robust regression loss? → MAE / Huber.
15. What's the output range of tanh? → (−1, 1).
