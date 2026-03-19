# Predicting Tipping in NYC FHVHV Trips at Scale using Spark


---

## Project Introduction

**Motivation and significance**

This project predicts whether a passenger will leave a tip (binary: tip > 0 vs no tip) using NYC For-Hire Vehicle High-Volume (FHVHV) trip data. The dataset includes trip attributes (miles, time, fare, surcharges), pickup/dropoff location IDs, time-derived features (hour, day of week, month), and categorical flags (e.g., shared request, WAV). Predicting tip behavior is useful for drivers and platforms to anticipate earnings and for understanding factors that influence tipping in ride-hail settings.

**Broader impacts and potential applications**

A reliable tip-prediction model could support driver decisions, platform incentives, and studies of fairness and demand. Accuracy matters for any downstream use (e.g., ranking trips or allocating resources) and for drawing valid conclusions about which features drive tipping.

**Why this problem required big data and distributed computing**

The full dataset has hundreds of millions of trips (e.g., ~745M rows before sampling). Fitting preprocessing, SVD on a dense feature matrix, and large-scale supervised learning on one machine would be impractical. We used **PySpark** end to end: DataFrames for EDA and feature pipelines, `RowMatrix.computeSVD` for unsupervised dimensionality reduction, then supervised models on the reduced representation, with splits and intermediate matrices written to Parquet. Memory-heavy jobs were run on a Spark cluster (e.g. multiple cores and executors, scratch-backed `spark.local.dir`).

---

## Notebooks and code

Exploration, preprocessing, and final modeling are implemented in the following notebooks 

- [EDA Notebook](https://github.com/javageek2018/SparkGroupProject/blob/main/eda_232r.ipynb)
- [First Model Notebook](https://github.com/javageek2018/SparkGroupProject/blob/main/preprocess_232r.ipynb) — preprocessing, vectorized features, **Model 1** (Random Forest)
- [Final Model Notebook](https://github.com/javageek2018/SparkGroupProject/blob/main/final_model_232r.ipynb) — **SVD**, reduced-feature classifiers, XGBoost and threshold analysis

---

## Methodology

### Data Exploration

Data were loaded from cleaned FHVHV Parquet. The deduplicated dataset has **745,273,321 rows**. We dropped identifiers not used as features (`originating_base_num`, `on_scene_datetime`, `dispatching_base_num`, `request_datetime`). Key columns include trip attributes (`trip_miles`, `trip_time`, `base_passenger_fare`, `tolls`, `tips`), location IDs (`PULocationID`, `DOLocationID`), and categorical flags. The target is binary: `tip_binary = 1` if `tips > 0`, else `0`. Class balance: **628,436,051 no-tip (0)** and **116,837,270 tip (1)** (~84% / ~16%). EDA included class counts and distributions of numeric and categorical features (e.g. `wav_match_flag`: Unknown, N, Y counts).

```python
df_clean = spark.read.parquet("my_cleaned_data/fhvhv_dedup")
df = df_clean.drop("originating_base_num", "on_scene_datetime", "dispatching_base_num", "request_datetime")
df1 = df.withColumn("tip_binary", F.when(F.col("tips").cast("double") > 0, 1).otherwise(0))
print("Rows:", df3.count())  # 745273321
df3.groupBy("tip_binary").count().show()
```

### Preprocessing (using Spark)

**Imputation:** Missing `airport_fee` and `congestion_surcharge` were filled with 0 (missing fee implies the charge did not apply). Null/empty `wav_match_flag` were set to `"Unknown"`. Remaining numeric variables (`log_fare`, `log_miles`, `log_minutes`, `tolls`, and engineered features) were imputed using Spark’s `Imputer` with the **median** strategy (appropriate for skewed financial variables).

**Scaling and transforms:** Log transforms (`log1p`) were applied to `base_passenger_fare`, `trip_miles`, and `trip_minutes` to reduce right skew. All numeric features were standardized with `StandardScaler` (withMean=True, withStd=True).

**Feature engineering:** Time-derived and derived flags: `pickup_hour`, `pickup_dow`, `pickup_month`, `is_weekend`, `is_night`, `has_airport_fee`, `has_congestion_fee`, `total_surcharges`, `fare_per_mile`.

**Encoding:** Six categorical variables—`hvfhs_license_num` (service provider), `shared_request_flag`, `shared_match_flag`, `access_a_ride_flag`, `wav_request_flag`, `wav_match_flag`—were trimmed, filled with `"Unknown"` where null/empty, then passed through `StringIndexer` and `OneHotEncoder` (handleInvalid="keep") to produce `*_ohe` columns.

**Assembly:** A final `VectorAssembler` combined `num_scaled` and all `*_ohe` into a single **39-dimensional** feature vector (15 scaled numeric + 24 OHE dimensions). The label is `tip_binary` cast to int as `label`. The result was written to `processed_data/vectorized_features`. For modeling, we took a 30% stratified sample by `label`, added a class-weight column (`balancing_ratio` ≈ 5.38 for the positive class), and split 80/20 into `train_df` and `test_df`, persisting both to Parquet.

```python
assembler_all = VectorAssembler(
    inputCols=["num_scaled"] + [f"{c}_ohe" for c in cat_cols],
    outputCol="features", handleInvalid="keep"
)
df_final = assembler_all.transform(df_enc).select(
    F.col("tip_binary").cast("int").alias("label"), "features", ...
)
df_model = df_model.sampleBy("label", fractions={0: 0.3, 1: 0.3}, seed=42)
train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=42)
train_df.write.mode("overwrite").parquet("processed_data/train_split")
test_df.write.mode("overwrite").parquet("processed_data/test_split")
# Modeling stage may read splits from Parquet paths as deployed on your cluster
```

### Model 1 (First Distributed Model)

Model 1 is a **Random Forest** classifier trained on the full 39-dimensional `features` vector with a `class_weight` column to handle class imbalance (inverse frequency weight ≈ 5.38 for the positive class). It was fit on `train_df` (30% stratified sample) and evaluated on `test_df`. Two configurations were compared: **RF1** (numTrees=20, maxDepth=10, maxBins=16) and **RF2** (numTrees=10, maxDepth=15, maxBins=12). RF2 achieved slightly higher ROC-AUC and PR-AUC with a minimal train–test gap.

```python
df_model = df_model.withColumn(
    "class_weight",
    when(col("label") == 1, balancing_ratio).otherwise(1.0)
)
rf = RandomForestClassifier(..., weightCol="class_weight")
model1 = rf.fit(train_df)
```

### Model 2 (SVD + supervised models)

**Data load:** `train_df` and `test_df` are read from saved Parquet splits (`train_split` / `test_split`).

**(1) SVD:** `RowMatrix` on training `features` (mllib vectors); `computeSVD(k=20, computeU=True)` with k = min(20, 39). **V** is 39×20; **s** singular values saved to `processed_data/svd_V` and `processed_data/svd_s` (explicit Parquet schema with Python floats). Explained-variance plots from **s** appear in Results/Figures.

**(2) Projection:** **V** loaded from Parquet, reshaped column-major (`order='F'`), broadcast; UDF projects each row: `np.dot(features.toArray(), V)` → 20-D `features`. Applied to train and test → `train_reduced_df`, `test_reduced_df` with `label` and `class_weight`.

**(3) Logistic Regression (Model 2a):** `LogisticRegression(featuresCol="features", labelCol="label", weightCol="class_weight", maxIter=10)` fit on `train_reduced_df`, evaluated on `test_reduced_df`. Model saved to `processed_data/lr_model_svd_20`. Coefficient magnitudes rank influential SVD directions (e.g. PC_1, PC_17, PC_13, PC_19, PC_11).

**(4) XGBoost on reduced sample (Model 2b):** 20% random samples of reduced train/test written to `processed_data/train_small_sample` and `test_small_sample` (~35.7M / ~8.9M rows). **SparkXGBClassifier** (`xgboost.spark`) with `num_workers=8`, binary logistic objective; initial run ~100 rounds (max_depth 6, learning_rate 0.1). **Hyperparameter grid:** four configs varying `max_depth`, `n_estimators`, `learning_rate`; best by PR-AUC: max_depth=10, n_estimators=100, learning_rate=0.05. **Threshold sweep** on best model’s positive-class probability on `test_small`; threshold **0.15** chosen by max F1 → reported TP/TN/FP/FN.

```python
features_rdd = train_df.select("features").rdd.map(lambda row: ml_to_mllib(row.features))
svd = RowMatrix(features_rdd).computeSVD(20, computeU=True)
# ... save V, s; project with broadcast V + UDF
lr = LogisticRegression(weightCol="class_weight", maxIter=10)
lr_model = lr.fit(train_reduced_df)
# XGBoost: SparkXGBClassifier(...).fit(train_small); param grid + threshold on prob_array[1]
```

---

## Results


### Data Exploration Results

The deduplicated sample has **~745M** rows with **~84%** no-tip and **~16%** tip (**class imbalance**). Tip **amounts** are highly skewed among tippers, supporting a **binary** label. **Tip rate** varies by **hour** and **trip distance**, and is higher for **airport-related** trips than others.

**Figure 1.** *Class imbalance.* Most trips have no tip; motivates class weighting and PR-AUC.

![Figure 1 — class balance](images/figure1.png)

**Figure 2.** *Tip amounts.* Strong right skew; binary tip/no-tip is a practical target.

![Figure 2 — tip amount distribution](images/figure2.png)

**Figure 3.** *Time of day.* Tip rates vary by pickup hour (e.g. midday / early afternoon).

![Figure 3 — tip rate by hour](images/figure3.png)

**Figure 4.** *Trip distance.* Higher tip rates on short-to-medium trips; more variability at long distances.

![Figure 4 — tip rate vs distance](images/figure4.png)

**Figure 5.** *Airport context.* Airport-related trips show higher tip rates.

![Figure 5 — airport vs non-airport](images/figure5.png)

### Preprocessing Results

After preprocessing: **39-dimensional** feature vector per row (15 scaled numeric + 24 OHE dimensions). Full dataset 745M rows; after 30% stratified sample ~223M rows; after 80/20 split, train and test sizes and class counts (e.g. train: ~178M no-tip, ~33M tip; test: ~44M no-tip, ~8M tip—adjust with your actual counts). Final feature matrix written to `processed_data/vectorized_features`.

### Model 1 Results

Random Forest (RF1 and RF2) metrics from the preprocessing notebook:

| Model | numTrees | maxDepth | maxBins | Train ROC-AUC | Test ROC-AUC | Train PR-AUC | Test PR-AUC | Gap   |
|-------|----------|----------|---------|---------------|--------------|--------------|-------------|-------|
| RF1   | 20       | 10       | 16      | 0.6443        | 0.6443       | 0.2427       | 0.2425      | ~0.00 |
| RF2   | 10       | 15       | 12      | 0.6530        | 0.6526       | 0.2481       | 0.2476      | 0.0004|


### Model 2 (SVD + supervised) Results

The original feature space consisted of **39 dimensions**, which were compressed into **20 principal components** via SVD. Our analysis of the singular values revealed several key insights:

* **Information Retention:** The 20 selected components capture **100.00% of the total variance**, which tells us that the structure of the dataset is fully preserved despite the reduction in dimensionality.
* **Feature Redundancy:** The ability to retain full variance while nearly halving the feature count implies significant multicollinearity within the raw taxi data. We were able to reduce this redundancy without sacrificing variance capture. We could potentially reduce the dimensionality further based solely on looking at our visualization.
* **Component Contribution:** The projection is heavily influenced by specific factors more than others; the first component (**PC_1**) exhibited the highest loading at **-0.54**, followed by **PC_17 (0.28)** and **PC_13 (-0.27)**.

**Figure 6.** *SVD explained variance.* Early components carry most variance (~90–95% in the leading directions shown); supports **k = 20** for downstream models.

![Figure 6 — SVD explained variance](images/figure6.png)

**Figure 7.** *First two SVD components.* Partial separation between tipped and non-tipped trips in the plane of the top two singular directions.

![Figure 7 — first two SVD components](images/figure7.png)

**Logistic Regression (full reduced train/test):**

A Logistic Regression model was implemented as a baseline to evaluate the discriminative power of the SVD-transformed features.

* **Performance Metrics:** The model achieved a **Test Accuracy of 60.01%** and an **Area Under the ROC Curve (AUROC) of 0.6368**.
* **Analysis:** While the baseline outperformed a random classifier, its performance was limited. This suggests that the relationship between the features and tipping behavior is likely non-linear, making it difficult for a purely linear estimator to capture tipping behavior.

| Metric          | Test (reduced) |
|-----------------|----------------|
| Accuracy        | 0.6001         |
| Area Under ROC  | 0.6368         |

**LR coefficient emphasis:** PC_1 (−0.54), PC_17 (0.28), PC_13 (−0.27), PC_19 (−0.27), PC_11 (0.26).

**XGBoost (Spark) on 20% sample of reduced features** (`train_small` / `test_small`):

To capture more complex feature interactions, an XGBoost classifier was trained on a 20% stratified sample of the reduced SVD features.

* **Comparative Performance:**
    * **Train ROC-AUC:** 0.6590 | **Test ROC-AUC:** 0.6584
    * **Train PR-AUC:** 0.2552 | **Test PR-AUC:** 0.2541
* **Generalization and Robustness:** The negligible difference between training and testing metrics (less than 0.1%) indicates **great generalization**. However, this also indicates we should continue focusing on hyperparameter tuning and/or feature engineering to improve performance.

| Setting | Train ROC-AUC | Test ROC-AUC | Train PR-AUC | Test PR-AUC |
|---------|-----------------|--------------|--------------|-------------|
| Initial (depth 6, 100 rounds, lr 0.1) | ~0.659 | ~0.658 | ~0.255 | ~0.254 |

**Hyperparameter grid** (same sample): best by PR-AUC — max_depth **10**, n_estimators **100**, learning_rate **0.05** → Test **ROC-AUC ~0.676**, **PR-AUC ~0.274**. Other runs in the grid: e.g. (8,150,0.05) ROC ~0.669 / PR ~0.265; (10,150,0.03) ROC ~0.675 / PR ~0.273; (8,200,0.03) ROC ~0.665 / PR ~0.261.

### Predictions analysis (XGBoost, test_small, probability threshold)

Threshold sweep on positive-class probability; **F1-maximizing threshold 0.15** (notebook output):

| Category        | Count   |
|-----------------|---------|
| True Positives  | 922,962 |
| True Negatives  | 4,290,456 |
| False Positives | 3,253,688 |
| False Negatives | 477,963 |

At 0.15: precision ~0.221, recall ~0.659, accuracy ~0.583, F1 ~0.331 (see notebook threshold table for other thresholds).

---

## Discussion

**Interpretation and “why”**

**Dimensionality reduction.** We used SVD via `RowMatrix.computeSVD` as the required unsupervised step. With **k = 20**, we keep the 20 strongest singular directions from the original **39**-dimensional feature space.

**Preprocessing.** Choices match the data shape: fee-like fields missing when not charged were filled with **0**; other skewed numerics used **median** imputation. **Log** transforms reduced skew on fare, miles, and trip time; **StandardScaler** put numeric features on comparable scales before assembly and SVD.

**Class imbalance.** About **84%** of trips are no-tip, so we used **inverse-frequency class weights** so models do not ignore the minority (tip) class.

**Model 1**

**Why Random Forest.** It scales well on Spark and is easy to interpret. We applied **class weights** (positive class ≈ **5.38×**) so the forest does not overwhelmingly favor the no-tip majority.

**Generalization vs. capacity.** Train and test **ROC-AUC** and **PR-AUC** are almost the same (gaps on the order of **0.0002–0.0004**). That points to stable generalization but also **underfitting**: the forests are probably too shallow or too small to fully exploit the signal.

**RF1 vs. RF2.** **RF2** (deeper trees, fewer trees than RF1) gained about **0.008** in test ROC-AUC over **RF1**, which suggests **depth** helps capture feature interactions for this task.

**Model 2 and dimensionality reduction**

**What we tested here.** After SVD, each trip is described by **20 numbers** instead of 39. We trained a **logistic regression** on those 20 numbers (using the full training set and scoring on the full test set). It gets **60%** of test rows’ labels right (**accuracy**) and separates tippers from non-tippers modestly well (**AUROC 0.64**; random guessing would be 0.5).

**How that compares to Model 1.** Model 1’s **RF2** used **all 39** original features (no SVD). Its ranking score on the same kind of test data was a bit higher (**ROC-AUC ~0.65**). So: **the simpler model on fewer, SVD-compressed features scores slightly worse than the random forest on the full feature set**—which is a common tradeoff (less information + a straight-line decision rule vs. more information + flexible tree rules).

**Boosted trees on reduced features.** **XGBoost** was trained on a **20% sample** of the reduced data. After tuning, test **ROC-AUC ~0.676** and **PR-AUC ~0.27** sit closer to RF2. That suggests **nonlinear** models on SVD coordinates can recover much of the signal. When comparing LR, XGB, and RF, remember **XGB used a smaller subset** of rows than LR/RF on their respective splits.

**Interpreting SVD directions.** Largest **LR** weights fell on **PC_1** and **PC_17**–**PC_19**. To tie those back to fare, time, zone, etc., you need the columns of **V** (how each original feature loads on each component).

**Fitting analysis**

**Random forests (Model 1).** Training and test scores are almost the same. That usually means the model **is not memorizing** the training data—but it also suggests the forests may be **too small or shallow** to squeeze out more performance (**underfitting**).

**XGBoost on the 20% sample.** Before tuning, train and test ROC-AUC were nearly identical (~**0.659** vs ~**0.658**), so the model was **not badly overfitting** on that slice. Trying different depth/learning-rate settings **nudged PR-AUC up** on the test slice (helpful when the positive class is rare).

**Takeaway.** **Logistic regression on SVD** is easy to read and good as a reference. **Tree boosting on the same 20-D SVD features** adds curved, interaction-like patterns **without** going back to all **39** raw features.

**Shortcomings**

- **One random split only.** We did not use k-fold or repeated splits, so metrics could shift a bit with another split.
- **Time and place.** Trips are not split by date or neighborhood; patterns might leak across train and test if similar trips cluster in time or space.
- **Not the full database.** Pipelines use a **30% stratified sample** of the full data, so results are for that slice, not every row in the raw release.
- **Features we don’t have.** No rider ID, trip history, or driver traits—only trip-level fields—so the ceiling on prediction may be limited.

---

## Conclusion

**Second model vs first**

**Model 2 (SVD + supervised).** We first reduced **39** features to **20** with **SVD**. On top of that we ran (1) **weighted logistic regression** on the full reduced train/test set—test **AUROC ~0.64**, **accuracy ~60%**—and (2) **XGBoost** on a **20% subsample** of reduced rows, with hyperparameter tuning—about **0.68** ROC-AUC and **0.27** PR-AUC on that subsample’s test portion.

**Model 1.** **RF2** on **all 39** features without SVD still lands around **0.65** ROC-AUC on its test set—so the best tree model on full features and the tuned boosted model on SVD features are in a similar ballpark, while plain LR on 20 SVD directions is a bit behind.

**What to try next**

- Try a **different number of SVD components** (not only 20).
- Run **boosting on all reduced-dimension rows** (not only 20%) if the cluster can handle it.
- Adjust **thresholds and class weights** depending on whether you care more about catching tippers (**recall**) or precision.
- Use **time- or zone-based splits** to check that results hold when the test period or area is new.

**Big data: how the work actually went**

The project starts from on the order of **745 million** trips—far more than a single machine can comfortably clean, featurize, and model end-to-end. **Spark** was the turning point: the same pipeline we would have wanted on a laptop (dedupe, aggregate, impute, scale, encode, assemble vectors, split, SVD, train classifiers) could run **across many workers**, so the problem became tractable instead of impossible.

Once we were on a cluster, the **shape of the work** changed. Heavy steps—wide shuffles on deduplication and group-bys, training random forests, computing SVD—were no longer “wait overnight and hope”; they became jobs we could **parallelize**. When RAM still wasn’t enough, pointing **temporary and shuffle data** at fast **scratch** storage let stages spill to disk instead of crashing. So distributed computing didn’t remove engineering effort; it moved the bottleneck to **configuration** (executors, memory, **shuffle partitions**) and to **stability** (avoiding OOM).

We also learned to **treat intermediates as assets**. Writing **vectorized features**, **train/test splits**, and **SVD factors** to **Parquet** meant we paid the cost of the hardest transforms once and could iterate on models without rebuilding everything from raw data. That pattern—compute once, reuse many times—is as important as the algorithms themselves at this scale.

**Where this could go next** follows naturally from that setup: run **boosting on all reduced-dimension rows** (not only a subsample), try **GBT** or stronger **imbalance-aware** objectives, add **zone/hour/provider** aggregates, use **clustering** on the SVD space for segmentation, train on **more than 30%** of the data if resources allow, and stress-test with **time- or geography-based splits** and **cross-validation** so scores reflect real deployment shifts.

---

## Statement of Collaboration

We worked as a **team of four**; each member made **substantial** contributions to the project.

- **Luigi Cheng — EDA, preprocessing, modeling support, writer:** Created **exploratory data analysis** visualizations; helped build the **data preprocessing** pipeline; assisted with **model tuning** for Model 1 and Model 2; wrote the **Introduction** and **Figures** sections and contributed to **Methods** in the final report.

- **Sripriya Panchapakesan — Coder (XGBoost & SVD), writer, submitter, feedback coordinator:** Implemented and ran the **XGBoost** and **SVD** parts of the modeling pipeline; drafted and revised the **written report**; handled **submission** logistics; **gathered feedback** from teammates and folded it into the report and workflow.

- **Cameron Hensley - Collaborater,Coder, writer:** Collaborated on model tuning for both the first and second model and performed code auditing on preprocessing and dimensionality reduction to improve optimization and verify logic. Contributed to report writing for third and fourth phase submissions as well.

- **Alex Twoy — Coordinator:** Coordinated team meetings and ensured that the group stayed on track and assigned roles for the final part; submitted the Abstract on the group's behalf at the beginning of the quarter; did Question 3 on Milestone 2 and performed the fitting analysis and conclusion on Milestone 4.
  
---


