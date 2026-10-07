# Data Science Case Study Questions (Round 2)

> **How to practice**: set a **25–30 min timer**, solve on paper *out loud* using the framework, then compare with the sample solution. The sample solutions are *one good answer*, not the only answer — interviewers value **structure, clarifying questions, and trade-offs** more than a "correct" answer.

## Contents
- [Part A — Frameworks (memorize these)](#part-a--frameworks)
- [Part B — Product Metrics & Root-Cause Cases](#part-b--product-metrics--root-cause-analysis-cases)
- [Part C — Machine Learning Modeling Cases](#part-c--machine-learning-modeling-cases)
- [Part D — Experimentation / A/B Testing Cases](#part-d--experimentation--ab-testing-cases)
- [Part E — Business Strategy & Guesstimate Cases](#part-e--business-strategy--guesstimate-cases)
- [Part F — Extra Practice Cases (no solutions — try yourself)](#part-f--extra-practice-cases)
- [Part G — Common Mistakes & Useful Phrases](#part-g--common-mistakes--useful-phrases)

---

## Part A — Frameworks

### A1. Universal case-study framework (use for every case)
```
1. CLARIFY      → Restate problem. Ask: goal? business model? users? time frame? constraints? data available?
2. STRUCTURE    → "I'll approach this in N steps…" (say it out loud)
3. METRICS      → Define success metric precisely (numerator, denominator, time window) + guardrails
4. HYPOTHESES / APPROACH → Break down (MECE), list hypotheses or modeling plan
5. DATA         → What data/features do I need? Where does it come from? Quality issues?
6. ANALYSIS / MODEL → Method, baseline, evaluation, validation
7. RECOMMEND    → Decision + expected impact + risks + next steps / how to measure (A/B test)
```

### A2. ML System Design framework (for "build a model to…" cases)
```
1. Business objective    → What decision will the model drive? What is the value?
2. ML problem framing    → Classification / regression / ranking / clustering? What is the label? Prediction horizon?
3. Data                  → Sources, label definition, leakage risk, class imbalance, size
4. Features              → User, item/product, behavior, time, context, aggregates
5. Model                 → Baseline (rule / logistic) → advanced (XGBoost / NN). Why?
6. Offline evaluation    → Metric tied to business cost, validation split (time-based!)
7. Deployment            → Batch vs real-time, latency, threshold choice
8. Online evaluation     → A/B test with business metric
9. Monitoring & iteration→ Drift, retraining, feedback loops, fairness
```

### A3. Metric-drop / Root-cause framework (for "metric X dropped 10%")
```
1. Clarify the metric   → Definition? How much drop? Sudden or gradual? Since when? Compared to what (WoW, YoY)?
2. Check data validity  → Tracking/logging bug? Pipeline delay? Definition change? Dashboard error?
3. External factors     → Seasonality, holidays, competitor launch, news, regulation, outages, weather
4. Internal factors     → Recent releases/features, marketing campaigns ended, pricing change, app update
5. Segment the drop     → Platform (iOS/Android/web), app version, geography, new vs returning, device, channel, user cohort
6. Decompose the metric → Metric = component × component (e.g., Revenue = Users × Conversion × AOV) — find which part dropped
7. Conclude & act       → Root cause + fix + how to prevent / monitor
```

### A4. Metric selection framework
- Use **AARRR funnel**: Acquisition → Activation → Retention → Referral → Revenue.
- Or **HEART** (Google): Happiness, Engagement, Adoption, Retention, Task success.
- Good metric = **measurable, sensitive, aligned with long-term value, hard to game**.
- Always propose: **1 North Star metric + 2–3 supporting metrics + guardrail metrics**.

### A5. Clarifying questions checklist
- What is the business goal (growth, revenue, cost reduction, user satisfaction)?
- Who are the users? What does the product do? How does the company make money?
- What decision will be made based on this analysis?
- What data do we have? How much history?
- Any constraints (latency, interpretability, regulation, budget)?
- What's the time frame?

---

## Part B — Product Metrics & Root-Cause Analysis Cases

### Case B1. Daily Active Users (DAU) of our app dropped 15% yesterday. Investigate.

**Clarify**
- How is DAU defined (opened app vs performed action)? Drop vs yesterday, last week same day, or forecast?
- Sudden (one day) or gradual? All markets or some?

**Step 1 — Check data issues first**
- Logging/tracking bug, new app release broke event firing, data pipeline delay (partial data), timezone change, definition change.

**Step 2 — External factors**
- Day of week / holiday / festival, major event (cricket final, elections), competitor launch, internet/cloud outage, app-store issue, news/PR crisis.

**Step 3 — Internal factors**
- New app version released? Server outage/latency increase? Login/OTP service failure? Push notifications stopped? Marketing campaign ended?

**Step 4 — Segment**
| Dimension | What a concentrated drop suggests |
|---|---|
| Platform (Android/iOS/web) | Platform-specific bug or release |
| App version | Bug in new version |
| Geography | Local outage, holiday, regulation |
| New vs returning users | New ↓ → acquisition/marketing issue; returning ↓ → product/engagement issue |
| Acquisition channel | Paid campaign stopped |
| Device / OS version | Compatibility bug |

**Step 5 — Decompose**: DAU = New users + Retained users + Resurrected users. Which component fell?

**Conclusion example**: "If the drop is only on Android version 5.2 released yesterday and concentrated on the login step, it's likely a login bug → roll back / hotfix, add monitoring alert on login success rate."

---

### Case B2. Revenue of an e-commerce company dropped 10% this month. Why?

**Decompose revenue**:
```
Revenue = Visitors × Conversion rate × Average Order Value (AOV)
AOV = Items per order × Avg price per item
```
- **Visitors ↓** → traffic source analysis (SEO, paid ads budget cut, app store ranking, seasonality).
- **Conversion ↓** → funnel analysis: home → product page → add to cart → checkout → payment. Find the step with the biggest drop (e.g., payment gateway failures ↑).
- **AOV ↓** → mix shift to cheaper categories, more discounts, fewer items per basket, removal of free-shipping threshold.

**Also check**: returns/cancellations ↑, refunds, specific categories/regions, stock-outs of best-sellers, price changes vs competitors, last year's same month (seasonality, e.g., festive season last year).

**Recommendation**: identify the component, quantify contribution of each factor ("70% of drop due to conversion drop in payments step on mobile web"), propose fix and monitoring.

---

### Case B3. How would you measure the success of a new feature — "Stories" on Instagram / "Reels" on a social app?

**Clarify goal**: increase engagement & time spent, content creation, retention, compete with competitor.

**Metrics**
- **Adoption**: % of DAU who view a story; % who post a story.
- **Engagement**: stories viewed per user per day, time spent on stories, completion rate, replies/reactions.
- **Creation**: stories posted per creator, % of users who are creators.
- **Retention**: D7/D30 retention of users who adopt vs not (careful: selection bias → use A/B test).
- **North star**: total time spent / meaningful interactions per DAU.
- **Guardrails / cannibalization**: feed posts, feed time, ad revenue (ads in feed), app performance (load time), unfollows/reports.

**Measurement**: A/B test with user-level randomization; watch for network effects (stories are seen by friends) → cluster randomization; novelty effect → run 3–4 weeks.

---

### Case B4. Uber/Ola — rider wait times increased in one city. Investigate.

- Clarify: average or P90 wait time? Since when? All areas or specific zones/times?
- **Supply side**: fewer active drivers (driver incentive change, fuel price hike, competitor offering better pay, festival), drivers concentrated elsewhere.
- **Demand side**: demand surge (event, rain, office reopening, metro strike).
- **Matching/tech**: dispatch algorithm change, map/GPS issues, ETA calculation bug.
- Segment: by zone, hour, car type, new vs experienced drivers.
- Metrics to look at: supply hours, requests, acceptance rate, cancellation rate, surge multiplier, driver utilization.
- Actions: targeted driver incentives in high-demand zones/times, surge pricing tuning, repositioning suggestions to drivers.

---

### Case B5. What metrics would you track for a food delivery app (Swiggy/Zomato)? Define a North Star.

- **North Star**: weekly orders delivered successfully (or GMV), balanced with customer retention.
- **Customer**: orders per user, repeat rate, D30 retention, conversion (app open → order), AOV, NPS/ratings, cancellation rate.
- **Delivery**: average delivery time, % on-time, delivery partner utilization, cost per delivery.
- **Restaurant**: # active restaurants, acceptance rate, prep time, restaurant churn.
- **Financial**: take rate, contribution margin per order, CAC, LTV/CAC.
- **Guardrails**: complaint rate, refund rate, food quality ratings.
- Mention trade-offs: faster delivery ↔ cost; discounts ↑ orders but ↓ margin.

---

### Case B6. Engagement on LinkedIn/our platform is increasing, but the number of new posts is decreasing. What's happening?

- Clarify definitions: engagement = likes/comments/time? posts = original posts only?
- Hypotheses:
  - Fewer creators but popular ones getting more engagement (concentration) → check posts per creator, creator count.
  - Feed algorithm change promoting viral/older content or videos.
  - Shift from posting to commenting/resharing (new feature).
  - Spam filter change removing posts.
  - Seasonality (holiday → people consume more, create less).
- Segment by creator type, region, content type.
- Risk: long-term health — consumption depends on creation; declining creators hurts future engagement. Recommend: creator incentives, notifications on post performance, track creator retention.

---

### Case B7. Funnel analysis — a fintech app's loan application completion rate dropped. How do you analyze?

- Map the funnel: app open → loan page → eligibility check → KYC upload → bank details → submit → approval → disbursal.
- Compute conversion at each step; compare with previous period; find the step with the largest drop.
- Segment by device, app version, region, user credit profile, channel.
- Check: KYC verification provider downtime, new mandatory field added, OTP failures, regulation change, interest rate change.
- Qualitative: session recordings, customer support tickets, user surveys.
- Fix and validate with A/B test (e.g., simplified KYC flow).

---

## Part C — Machine Learning Modeling Cases

### Case C1. Build a customer churn prediction model for a telecom / subscription company.

**1. Business objective**
Identify customers likely to churn in the next 30 days so the retention team can offer incentives. Goal: reduce churn rate while controlling incentive cost.

**2. Problem framing**
- Binary classification. **Label**: churned = didn't renew / cancelled within next 30 days after the snapshot date.
- Define churn clearly (contractual: cancellation; non-contractual like e-commerce: no purchase in 90 days).
- Prediction made monthly for all active customers (batch).

**3. Data & features**
| Category | Features |
|---|---|
| Demographics | Age, location, tenure |
| Account | Plan type, contract length, payment method, price, discounts |
| Usage | Calls/data usage trends (last 7/30/90 days), % change in usage, last active date |
| Engagement | Logins, app usage, feature usage |
| Support | # complaints, tickets, resolution time, network issues |
| Billing | Late payments, bill shock, price increases |
| Competitive | Competitor offers in region |
- Strong signals: **usage declining trend**, complaints, contract ending soon.
- **Avoid leakage**: only use data available before the snapshot date (e.g., not "cancellation request call").

**4. Model**
- Baseline: logistic regression (interpretable).
- Main: XGBoost/LightGBM; handle imbalance (churn ~5–20%) with class weights.
- Time-based train/validation split (train on Jan–Jun snapshots, validate on Jul).

**5. Evaluation**
- PR-AUC, recall at top-K% (retention team can call only top 10%), lift chart (top decile lift).
- Business: choose threshold where expected profit is max:
  `Profit = Σ [P(churn) × P(saved | offer) × CLV − cost of offer]`

**6. Beyond prediction — uplift modeling**
- Predicting churn ≠ knowing who will respond to an offer. Segment users into: persuadables, sure things, lost causes, sleeping dogs. **Uplift models** target persuadables.

**7. Deployment & measurement**
- Monthly batch scores to CRM. Explain reasons using SHAP ("usage dropped 40%, 3 complaints").
- A/B test: treatment group gets retention offers based on model, control gets business-as-usual → measure churn rate difference and ROI.
- Monitor drift, retrain monthly/quarterly.

---

### Case C2. Design a credit card fraud detection system.

**Clarify**: real-time at transaction time? Cost of FN (fraud loss) vs FP (blocked genuine customer → bad experience)? Latency limit (~100ms)?

**Framing**: binary classification, extremely imbalanced (~0.1% fraud). Labels from chargebacks (delayed by weeks → label delay!).

**Features**
- Transaction: amount, merchant category, online/offline, time of day, country.
- **Behavioral/velocity**: # transactions in last 1 hr/24 hr, amount vs user's average, new merchant, new device, distance from last transaction (impossible travel), time since last txn.
- Device/IP: new device, IP risk score, VPN.
- Merchant risk: historical fraud rate of merchant.
- Graph features: shared devices/cards across accounts.

**Model**
- Rules engine (fast, interpretable) + ML model (XGBoost) + anomaly detection (Isolation Forest/autoencoder) for new fraud patterns.
- Imbalance: class weights, undersampling non-fraud, focus on PR-AUC.

**Decisioning (3 tiers)**
- Score < low threshold → approve.
- Middle → step-up authentication (OTP) — reduces friction cost of FP.
- High → block + manual review.
- Thresholds chosen by cost: minimize `FN × avg fraud amount + FP × customer friction cost` subject to review team capacity.

**Evaluation**: precision, recall, PR-AUC, $ fraud caught, false decline rate, latency.

**Challenges**: concept drift (fraudsters adapt) → frequent retraining, monitoring; adversarial behavior; label delay; feedback loop (blocked transactions have no label → use small random exploration/holdout).

---

### Case C3. Design a recommendation system for an e-commerce site / Netflix.

**Clarify**: Where are recommendations shown (home page, "similar items", email)? Goal: CTR, purchases, watch time, diversity? Scale (millions of users & items)?

**Data**: implicit feedback (views, clicks, add-to-cart, purchases, watch %), explicit (ratings), item metadata (category, price, brand, text, images), user profile, context (time, device).

**Architecture (2-stage)**
1. **Candidate generation** (millions → ~500): collaborative filtering (matrix factorization / ALS), item-to-item similarity ("bought together"), content-based embeddings, popular/trending, two-tower neural model + approximate nearest neighbor search.
2. **Ranking** (500 → top 20): gradient boosted trees or deep model predicting P(click/purchase) with user, item, user-item interaction & context features.
3. **Re-ranking**: diversity, freshness, business rules (in stock, remove purchased items), fairness.

**Cold start**
- New user: popular/trending items, onboarding preferences, location/demographics, session-based recs.
- New item: content features (text/image embeddings), exploration boost.

**Evaluation**
- Offline: Precision@K, Recall@K, NDCG, MAP, coverage, diversity (time-based split).
- Online A/B test: CTR, conversion, revenue per user, watch time, long-term retention; guardrails: latency, diversity.

**Pitfalls**: popularity bias, filter bubbles, feedback loops (recommendations influence future training data) → exploration (bandits).

---

### Case C4. Predict house prices / car prices for a listing platform.

- Framing: regression. Target: final sale price (log-transform because skewed).
- Features: area, bedrooms, bathrooms, age, floor, location (lat/long, locality, distance to metro/schools/offices), amenities, furnishing, neighborhood average price (**careful with leakage — compute from past data only**), listing text (NLP), images (CNN embeddings), market trends (time).
- Data issues: outliers (typos: 1,000,000 sq ft), missing values, duplicate listings.
- Models: linear regression baseline → Random Forest / XGBoost / LightGBM.
- Metrics: MAE (interpretable in ₹), MAPE (relative), RMSE; evaluate per segment (luxury vs budget).
- Validation: time-based split (prices change with time); spatial leakage — same building in train & test.
- Product: show price range (prediction intervals via quantile regression), explain drivers to users.

---

### Case C5. A bank wants to predict loan default (credit risk scoring).

- Framing: binary classification — default = 90+ days past due within 12 months.
- Features: credit bureau score, income, debt-to-income ratio, employment type/tenure, existing loans, repayment history, credit utilization, loan amount/tenure, banking transactions (salary credits, bounced cheques).
- Constraints: **regulation & explainability** → logistic regression scorecard (WOE binning) or XGBoost with SHAP explanations; **fairness** — no protected attributes (gender, religion), check proxy bias.
- **Reject inference**: we only have outcomes for approved applicants → biased training data.
- Metrics: AUC, Gini (2·AUC − 1), KS statistic, calibration (probability used for pricing), expected loss = PD × LGD × EAD.
- Decision: approval threshold chosen to balance approval rate vs default rate; risk-based pricing.
- Monitoring: PSI for drift, vintage analysis of defaults.

---

### Case C6. Forecast demand for products in a retail / quick-commerce (Blinkit/Zepto) warehouse.

- Clarify: granularity (SKU × store × day/hour), horizon (next 7 days), purpose (inventory replenishment), cost of stock-out vs overstock (perishables!).
- Features: historical sales lags (t−1, t−7, t−14), rolling means, day of week, holidays/festivals, weather, promotions/discounts, price, events, stock-out flags (sales = 0 because out of stock, not no demand → **censored demand**).
- Models: seasonal naive baseline → Prophet / SARIMA for top SKUs → global LightGBM model across SKUs → DL (TFT/DeepAR) at scale.
- Metrics: WAPE / MAPE, bias (over vs under forecasting), service level; asymmetric loss (quantile loss — forecast P80 for perishables trade-off).
- Validation: walk-forward backtesting.
- Business output: reorder quantities = forecast + safety stock.

---

### Case C7. Build a model to detect fake reviews / spam on a platform.

- Labels: manually labeled reviews, user reports, known spammer accounts; possibly weak labels.
- Features: text (TF-IDF, BERT embeddings, sentiment extremity, duplicate/near-duplicate text), reviewer behavior (account age, # reviews/day, all 5-star, reviews only one seller), product-level (sudden burst of reviews), network (groups of accounts reviewing same products, shared IP/device).
- Models: text classifier + behavioral GBM + graph-based detection (collusion rings).
- Metrics: precision (falsely removing genuine reviews hurts trust) + recall; human review queue for borderline cases.
- Adversarial → continual retraining, monitoring.

---

### Case C8. Customer segmentation for a marketing team.

- Goal: personalized campaigns.
- Data: RFM (Recency, Frequency, Monetary), product categories bought, channel preference, demographics, engagement with emails, discount sensitivity.
- Approach: feature scaling → K-Means / GMM (or RFM rule-based segments as baseline), choose k using silhouette + business interpretability (4–6 segments).
- Profile segments: "High-value loyalists", "Discount seekers", "At-risk churners", "New users", "Dormant".
- Action per segment + measure via A/B test of targeted vs generic campaign.
- Re-cluster periodically; track segment migration.

---

### Case C9. Design a system to auto-classify customer support tickets (NLP / LLM).

- Goal: route tickets to right team, reduce resolution time.
- Data: historical tickets with assigned category; class imbalance; messy multilingual text.
- Approach: baseline TF-IDF + logistic regression → fine-tuned BERT/DistilBERT → or LLM zero/few-shot classification with RAG over help docs to also draft replies.
- Metrics: macro-F1, accuracy per category, % auto-routed correctly; business: first-response time, resolution time, CSAT.
- Deployment: confidence threshold — low confidence goes to human; human corrections feed retraining.
- LLM-specific: hallucination guardrails, PII masking, cost/latency per ticket.

---

### Case C10. Predict the ETA for a food delivery order.

- Components: restaurant prep time + driver assignment time + pickup travel time + delivery travel time.
- Features: restaurant historical prep time by hour, order size, current restaurant load, distance, traffic, weather, driver location/availability, time of day, day of week.
- Model: regression (LightGBM) per component or end-to-end; quantile regression for P90 (promise times).
- Metric: MAE, % orders within ±5 min, **asymmetric cost** — late is worse than early (customer dissatisfaction) → tune bias.
- Online: A/B test impact on conversion (shown ETA affects ordering!), cancellation, ratings.

---

## Part D — Experimentation / A/B Testing Cases

### Case D1. We changed the "Buy Now" button color. Conversion went from 10% to 10.5%. Should we launch?

**Clarify**: Sample size per group? Test duration? Was it pre-registered?

**Analysis**
- Two-proportion z-test. Suppose n = 20,000 per group:
  - p_pool = 0.1025, SE = √(0.1025×0.8975×(2/20000)) ≈ 0.00303
  - z = 0.005/0.00303 ≈ 1.65 → p ≈ 0.10 (two-tailed) → **not significant** at α=0.05.
- Check power: to detect 0.5% absolute lift at 10% baseline need n ≈ 16×0.09/0.000025 ≈ **57,600 per group**. Test is underpowered.
- Check SRM, duration ≥ 1 week, novelty effect, segments, guardrails (revenue, AOV).

**Recommendation**: continue running until planned sample size is reached; don't launch yet based on this data. Also: practical significance — 0.5% lift on large traffic could be valuable; low cost change.

---

### Case D2. Design an A/B test for a new recommendation algorithm on an e-commerce homepage.

1. **Hypothesis**: new algorithm increases purchases per user.
2. **Metrics**: primary — revenue per user or conversion rate; secondary — CTR on recs, items per order; guardrails — page latency, return rate, diversity, bounce rate.
3. **Unit**: user (logged-in) → consistent experience.
4. **Sample size**: from baseline conversion & variance, MDE (e.g., 1% relative), α=0.05, power 0.8; revenue metric is high-variance → use CUPED or trim outliers.
5. **Duration**: ≥ 2 weeks (weekly cycles), avoid sale events.
6. **Ramp-up**: 1% → 5% → 50% to catch bugs.
7. **Analysis**: t-test on revenue per user, CI, segment by new/returning users, device.
8. **Pitfalls**: novelty effect, interference minimal here, multiple metrics → correct.
9. **Decision**: launch if primary improves significantly with no guardrail violations and cost is justified.

---

### Case D3. A/B test shows the treatment increased clicks but decreased revenue. What do you do?

- Verify data (SRM, bugs, outliers).
- Understand mechanism: clickbait-y recommendations? Users clicking more but finding less relevant items? Cheaper items promoted?
- Look at full funnel: click → add to cart → purchase; AOV; segments.
- Decide based on company's **primary goal (OEC)** — usually revenue/long-term value beats clicks. Clicks are a proxy; when proxy and true goal disagree, trust the true goal.
- Recommend: don't launch; iterate on the algorithm (optimize for purchase probability, not CTR).

---

### Case D4. How would you test a new pricing strategy for a ride-sharing app?

- Problem: user-level randomization in marketplaces causes **interference** — treatment pricing changes driver supply available to control users; also fairness concerns (two users seeing different prices at same location — PR risk).
- Options:
  - **Geo-experiment**: randomize cities/regions (fewer units → lower power; use matched markets, synthetic control).
  - **Switchback test**: alternate pricing for the whole city by time windows (e.g., every 30 min, randomized).
- Metrics: rides completed, revenue, driver earnings, rider conversion, wait time, cancellation, long-term rider retention.
- Analysis: account for autocorrelation in time windows; run several weeks.

---

### Case D5. You can't run an A/B test (e.g., a TV ad campaign launched nationally). How do you measure its impact?

- **Pre-post with control**: Difference-in-Differences if campaign ran in some regions only.
- **Synthetic control**: build a weighted combination of non-exposed regions to mimic the treated one.
- **Interrupted time series / causal impact (Bayesian structural time series)**: forecast what would have happened without the campaign, compare to actual.
- **Marketing Mix Modeling (MMM)**: regression of sales on spend across channels with adstock & saturation effects.
- Check confounders: seasonality, competitor activity, price changes.

---

### Case D6. The A/B test result is significant overall, but negative for mobile users and positive for desktop. What do you do?

- Check if segment sizes are adequate (small segments → noisy; multiple comparisons).
- Check for Simpson's paradox and whether randomization was balanced across devices.
- Investigate a mobile-specific bug/UX issue (layout, load time).
- Options: launch on desktop only, fix mobile and re-test, or launch overall if mobile negative effect is small and not significant after correction.

---

## Part E — Business Strategy & Guesstimate Cases

### Case E1. Estimate the number of cups of tea/coffee sold daily in Bangalore.
```
Population ≈ 13 million
Adults (15+) ≈ 75% → ~10M
Tea/coffee drinkers ≈ 80% → 8M
Avg cups per day per drinker ≈ 2 → 16M cups
Bought outside (vs home) ≈ 30% → ~5M cups/day sold
```
State assumptions clearly, sanity check, and mention how you'd refine (survey, segment by office-goers/students).

### Case E2. Estimate the market size for an online grocery delivery app in a Tier-1 Indian city.
- Households → % with smartphones & internet → % who shop groceries online → orders/month → AOV → monthly GMV.
- Example: 4M households × 60% smartphone/online-capable × 25% adoption × 4 orders/month × ₹600 ≈ **₹144 crore/month**.

### Case E3. Swiggy wants to launch a subscription program (like Swiggy One). How would you evaluate it?
- Goals: increase order frequency & retention, lock-in vs competitors.
- Economics: subscription fee revenue + incremental orders × margin − cost of free deliveries/discounts − cannibalization (users who'd order anyway).
- Analysis: identify target users (frequent orderers vs medium — medium users have most uplift potential), price sensitivity (survey/price testing), break-even order frequency.
- Experiment: offer to random subset → measure incremental orders per user, retention, net margin over 2–3 months.
- Metrics: subscription uptake, renewal rate, orders/subscriber vs control, contribution margin, LTV.

### Case E4. Should an e-commerce company offer free shipping above ₹499 instead of ₹999?
- Trade-off: ↑ conversion & orders vs ↑ shipping cost and possibly ↓ AOV (users stop adding items to reach ₹999).
- Data: distribution of cart values (how many orders between ₹499–999), shipping cost per order, conversion lift estimates, cart abandonment due to shipping fees.
- Run A/B test (user-level) → primary metric: **net profit per user**; secondary: conversion, AOV, orders/user.
- Recommend based on profit, not just orders.

### Case E5. A company's marketing team wants to know which channel gives the best ROI.
- Define ROI = (incremental revenue − cost)/cost.
- Attribution: last-click (biased), first-click, linear, time-decay, data-driven (Shapley/Markov) attribution.
- Better: incrementality via holdout experiments (geo lift tests, ghost ads) + Marketing Mix Modeling.
- Metrics: CAC per channel, LTV of acquired users (quality matters, not just volume), payback period.

### Case E6. A bank sees rising credit card defaults. What would you analyze and recommend?
- Clarify time period, product, portfolio segment.
- Vintage analysis: are recent cohorts (newly approved) defaulting more? → approval policy/model drift.
- Segment: income band, region, channel (online vs branch), employment type, credit limit.
- External: macroeconomy (layoffs, inflation, interest rate hikes).
- Check model: PSI drift in applicant population, calibration of risk model.
- Actions: tighten approval cut-off for risky segments, lower limits, early warning model for collections, retrain risk model.

---

## Part F — Extra Practice Cases

> Solve these yourself using the frameworks. Time each one (25 min).

**Product / Metrics**
1. YouTube watch time dropped 5% week-over-week. Investigate.
2. How would you measure the success of WhatsApp's "Communities" feature?
3. Define metrics for Google Maps' "Live traffic" feature.
4. Number of Uber drivers increased, but completed rides stayed flat. Why?
5. Spotify: how would you decide whether to show a "Daily Mix" on the home screen?
6. App-store rating of our app dropped from 4.5 to 3.9 in a month. What's happening?
7. Average session time increased but retention decreased. Good or bad?
8. Amazon Prime sign-ups dropped after a price increase. How would you analyze the impact?

**Machine Learning**
9. Build a model to predict which customers will respond to a marketing email.
10. Predict hospital patient readmission within 30 days.
11. Design a search ranking system for an e-commerce site.
12. Detect fake accounts on a social media platform.
13. Predict the optimal price for a hotel room (dynamic pricing).
14. Build a model to predict employee attrition for HR.
15. Detect defective products from images on a manufacturing line.
16. Predict insurance claim fraud.
17. Build a lead-scoring model for a B2B sales team.
18. Build an LLM-powered chatbot that answers questions from company policy documents — how would you design and evaluate it?
19. Predict which users will convert from free trial to paid.
20. Predict the click-through rate of ads (CTR prediction).

**Experimentation**
21. Design an experiment to test whether adding video reviews increases sales.
22. You ran 10 variants of a landing page; one is significant at p=0.04. Launch it?
23. How would you A/B test a new feature in a social network where users interact?
24. An experiment's results are very different in week 1 and week 3. Why?

**Business / Guesstimates**
25. Estimate the number of ATMs in India.
26. Estimate monthly revenue of a popular café near your college.
27. Should Zomato enter the grocery delivery business?
28. How would you decide where to open the next store for a retail chain?
29. Estimate how many Ola/Uber rides happen in Mumbai per day.
30. Netflix is considering a cheaper ad-supported plan. How would you evaluate it?

---

## Part G — Common Mistakes & Useful Phrases

### ❌ Mistakes to avoid
- Jumping into a solution without clarifying questions.
- Choosing accuracy for imbalanced problems.
- Ignoring data leakage and time-based validation.
- Proposing deep learning first without a baseline.
- Forgetting the **business decision** and **how success is measured** (A/B test).
- Going silent — think aloud.
- Listing 20 hypotheses without prioritizing; prioritize by likelihood × impact × ease of checking.
- Ignoring trade-offs (cost of FP vs FN, latency, interpretability, fairness).

### ✅ Useful phrases
- "Before I dive in, let me ask a few clarifying questions…"
- "Let me structure my approach into four parts…"
- "I'd first rule out data/instrumentation issues, since that's the most common cause of sudden drops."
- "Let me define the metric precisely: numerator is…, denominator is…, over a 7-day window."
- "I'd start with a simple, interpretable baseline so we have a benchmark."
- "The key trade-off here is between precision and recall, because a false negative costs us X while a false positive costs Y."
- "To validate this in production, I'd run an A/B test with … as the primary metric and … as guardrails."
- "If I had more time, I'd also look at…"
- "To summarize: the recommendation is…, expected impact…, risks…, and next steps…"

### Final 60-second summary template (end every case with this)
> "To summarize, the problem was **[X]**. I'd measure success using **[metric]**. My approach is **[analysis/model]**, using **[key data/features]**. The main risks are **[leakage/bias/cost trade-off]**, which I'd handle by **[mitigation]**. I'd validate impact with **[A/B test / backtest]** and monitor **[drift/guardrails]** after launch."
