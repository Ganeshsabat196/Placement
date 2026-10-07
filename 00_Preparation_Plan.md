# Data Scientist Interview — Preparation Plan

Your interview has two rounds:

| Round | What it tests | Files to use |
|-------|---------------|--------------|
| **Round 1 – Concepts** | Statistics, probability, ML, deep learning, SQL, Python, evaluation metrics, A/B testing | `01_DS_Concepts_Notes.md`, `02_DS_Interview_Questions.md` |
| **Round 2 – Case Study** | Business sense, problem structuring, metrics, model design, experiment design, communication | `03_DS_Case_Study_Questions.md` |

---

## 1. How to use these files

1. **Read `01_DS_Concepts_Notes.md` first** – it is your theory book. Make sure you can explain each concept in 2–3 sentences *in your own words* and give one example.
2. **Then go through `02_DS_Interview_Questions.md`** – cover the answer, say your answer out loud, then compare. Mark questions you got wrong with ❌ and revisit them.
3. **Practice `03_DS_Case_Study_Questions.md`** – set a 25–30 minute timer per case, solve it on paper using the framework, *then* read the sample solution.
4. **Final 2 days** – only revise: the "Quick Revision Cheat Sheet" at the end of file 01, your ❌ questions, and 5 case studies.

---

## 2. Suggested 3-week schedule (compress to 1 week if needed)

### Week 1 – Foundations (Round 1)
| Day | Topic |
|-----|-------|
| 1 | Descriptive stats, probability, Bayes theorem, distributions |
| 2 | Inferential stats: CLT, hypothesis testing, p-values, confidence intervals, t/z/chi-square/ANOVA |
| 3 | A/B testing end-to-end (sample size, power, pitfalls) |
| 4 | Linear & logistic regression, assumptions, regularization, bias–variance |
| 5 | Trees, Random Forest, Boosting (XGBoost/LightGBM), SVM, KNN, Naive Bayes |
| 6 | Unsupervised: K-Means, hierarchical, DBSCAN, PCA; evaluation metrics |
| 7 | Revision + 40 questions from file 02 |

### Week 2 – Applied skills
| Day | Topic |
|-----|-------|
| 8 | Feature engineering, missing data, imbalanced data, data leakage, cross-validation |
| 9 | SQL: joins, GROUP BY, window functions, CTEs — solve 15 problems (LeetCode/StrataScratch) |
| 10 | Python/Pandas/NumPy — groupby, merge, apply, vectorization |
| 11 | Deep learning basics: NN, backprop, activation, optimizers, CNN, RNN/LSTM, Transformers |
| 12 | NLP & GenAI basics (TF-IDF, embeddings, BERT, LLMs, RAG), time series basics |
| 13 | Model deployment/MLOps basics, recommendation systems |
| 14 | Mock Round 1 – ask a friend to quiz you from file 02 |

### Week 3 – Case studies (Round 2)
| Day | Topic |
|-----|-------|
| 15 | Learn the case framework; product metric cases (DAU drop, engagement) |
| 16 | ML system design cases (churn, fraud, recommendation) |
| 17 | Experimentation cases (A/B test design, launch decisions) |
| 18 | Business/estimation cases (pricing, guesstimates, marketing ROI) |
| 19 | Explain your own resume projects as case studies (STAR + metrics) |
| 20 | 2 full mock case interviews (timed) |
| 21 | Light revision, rest |

---

## 3. What interviewers actually look for

**Round 1 (Concepts)**
- Clear, *intuitive* explanations — not memorized definitions.
- Knowing **when** to use something and its **trade-offs** (e.g., "Random Forest reduces variance, boosting reduces bias").
- Ability to go one level deeper on follow-ups ("Why does L1 give sparse solutions?").
- Connecting concepts to your projects.

**Round 2 (Case Study)**
- **Asking clarifying questions** before jumping in.
- **Structured thinking** – say your framework out loud.
- **Choosing the right metric** and defining it precisely.
- Practical modeling: data you need, features, baseline, evaluation, deployment, monitoring.
- **Business impact** – always end with "what decision does this drive and how do we measure success?"
- Mentioning trade-offs, risks, and edge cases.

---

## 4. Golden rules for answering

1. **Structure**: "I'd approach this in 3 parts: …"
2. **Define → Intuition → Example → Trade-off/When to use.**
3. If you don't know: say what you *do* know, reason from first principles, don't bluff.
4. Think aloud in case studies — silence is the biggest mistake.
5. Always **link to your projects** ("In my churn project, I used SMOTE because …").
6. Quantify impact on your resume projects (accuracy ↑, cost ↓, time saved).

---

## 5. Your resume projects — prepare this for each project

| Question | Your answer |
|----------|-------------|
| What was the business problem? | |
| What data did you use? Size? Source? | |
| How did you clean it / handle missing values? | |
| What features did you create? | |
| Which models did you try and why? Baseline? | |
| Which metric and why? | |
| Results (numbers)? | |
| What would you do differently / next? | |
| Biggest challenge and how you solved it? | |

---

## 6. Practice resources
- **SQL**: LeetCode Database (Top 50 SQL), StrataScratch, DataLemur
- **Stats/Prob**: "Practical Statistics for Data Scientists" (Bruce), Khan Academy statistics
- **ML**: "Hands-On ML" (Géron), StatQuest videos (Josh Starmer) — excellent for intuition
- **Case studies**: "Ace the Data Science Interview" (Huo & Singh), "Trustworthy Online Controlled Experiments" (Kohavi) for A/B testing
- **Mock interviews**: Pramp, peers, record yourself answering
