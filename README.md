# Self-Checkout Fraud Detection

Detecting fraudulent transactions at supermarket self-checkout terminals using a mix of supervised classifiers, unsupervised anomaly detection, and customer-behaviour clustering.

The motivation: self-checkout fraud (skipping items, mis-scanning expensive products as cheaper ones, abandoning items in the bagging area) is one of the largest sources of "shrinkage" in modern grocery retail. Labelled fraud data is scarce, so the project combines:

1. A well-known **labelled credit-card fraud benchmark** to validate that standard supervised classifiers behave as expected on extreme class imbalance.
2. **Unsupervised anomaly detection** (Isolation Forest, One-Class SVM) which is the realistic operating mode in stores where labels don't exist.
3. **Customer-behaviour clustering** on real per-transaction supermarket data, plus a pilot study with hand-labelled fraud cases to train a logistic regression fraud classifier.

## Why supervised learning is the right approach (per store)

A central finding of the pilots was that self-checkout fraud is **not generalisable across stores**. The customer base of a supermarket depends heavily on its location — a branch in a student area behaves very differently from a branch in a family-suburb, a city-centre commuter location, or a tourist district. Each customer base has its own:

- typical basket compositions (e.g. students buy mostly alcohol, snacks, ready meals; family branches buy bulk fresh produce + dairy; tourist-area branches skew towards small high-margin items),
- typical scan times and basket sizes,
- and consequently, its own fraud patterns — *which* items get mis-scanned or skipped depends on what's commonly in baskets in the first place.

This means an unsupervised "anomaly" model trained on one store will mark the *normal* shopping behaviour of customers at another store as fraudulent, and miss that store's actual fraud (which by definition isn't anomalous *there*). A single global fraud-detection model is therefore not a viable approach.

The conclusion is that **each store needs its own supervised fraud-detection model**, trained on its own labelled fraud cases against its own customer base. Supervised learning works precisely because the labels encode the store-specific definition of "fraudulent vs normal" — one that no globally-trained anomaly detector can recover. The unsupervised experiments in this repo (`main1.py`, `main2.py`, `main3.py`) exist primarily to demonstrate this gap: anomaly detectors flag novelty, not fraud, and at the per-store level the two diverge sharply.

The pilot logistic regression (`R_scripts/predict 1.Rmd`) is the prototype of this per-store supervised pipeline.

## Repository structure

```
FD/
├── scripts/                    # Python pipeline
│   ├── main.py                 # Supervised baselines on Kaggle creditcard.csv
│   ├── main1.py                # Isolation Forest on synthetic 2-D data (sanity check)
│   ├── main2.py                # Isolation Forest on creditcard.csv with label evaluation
│   └── main3.py                # Isolation Forest + PCA on supermarket transaction data
├── R_scripts/                  # R analysis on real supermarket pilot data
│   ├── Pilot_working (2) (1) (2).Rmd   # Feature engineering + k-means clustering (k=24)
│   └── predict 1.Rmd                   # Pilot 1 + Pilot 2 + logistic regression + ROC/AUC
└── data/
    ├── supermarket.csv         # Raw scan logs (full)
    ├── supermarket_1.csv       # Reduced supermarket dataset used by main3.py
    ├── supermarket_2.csv       # Pilot 2 raw
    ├── supermarket_2 (1).xlsx  # Pilot 2 (Excel)
    ├── supermarket_3.xlsx      # Pilot 1 raw (74 item slots)
    └── supermarket_final.xlsx  # Final cleaned/feature-engineered dataset
```

The Kaggle `creditcard.csv` (~150 MB, 284,807 transactions, 492 fraud cases) is **not** included in the repo because it exceeds GitHub's 100 MB file limit. Download it from <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> and place it under `data/`.

## Methods

### 1. Supervised baselines on credit-card fraud (`scripts/main.py`)

Standard pipeline applied to the Kaggle creditcard dataset:

- StandardScaler on `Amount`; drop `Time`; drop duplicates.
- 75/25 train/test split.
- Six classifiers compared by accuracy and F1:
  - Decision Tree (`max_depth=4`, entropy)
  - K-Nearest Neighbors (k=7)
  - Logistic Regression
  - Support Vector Machine (RBF)
  - Random Forest (`max_depth=4`)
  - XGBoost (`max_depth=4`)

Confusion matrices and F1 (rather than accuracy) are the relevant metric here because the positive class is ~0.17% of all transactions.

### 2. Isolation Forest sanity check on synthetic data (`scripts/main1.py`)

Two-cluster Gaussian "normal" data (mean ±2, σ=0.3) plus uniform-noise outliers in `[-4, 4]²`. The script trains `IsolationForest(contamination=0.1)`, then computes:

- Correct outlier detections,
- False positives (normals flagged as anomalies),
- Undetected outliers,

and visualises each. This is purely to verify that the Isolation Forest configuration behaves sensibly before applying it to noisy real data.

### 3. Isolation Forest on credit-card data with label evaluation (`scripts/main2.py`)

Same data prep as `main.py`, then:

- Fit Isolation Forest on the unsupervised feature matrix.
- Map sklearn's `{1, -1}` output to `{0, 1}` so it lines up with the dataset's `Class` labels.
- Compare predicted anomalies to the true fraud labels — counting overall correct detections and correct **fraud-class** detections specifically.

### 4. Isolation Forest + PCA on supermarket data (`scripts/main3.py`)

Per-customer features extracted from `supermarket_1.csv`, then `PCA(0.9)` (retain 90% variance) before fitting Isolation Forest with `contamination=0.1`. Predictions are again mapped to `{0,1}` and compared to the labelled `Class` column.

### 5. R: feature engineering + k-means clustering (`R_scripts/Pilot_working ...Rmd`)

Each row in `supermarket_3.xlsx` is one customer's full self-checkout session, encoded as triples of `(category, scan_time, price)` for up to 74 items. The R script:

- Splits the wide-format triples into per-item columns (`splitstackshape::cSplit`).
- Aggregates per-customer features:
  - `totalTime`, `totalItems`, `totalPrice`
  - `AvgTime`, `AvgPrice`
  - 18 department counts: `BakeryPastry`, `BeerWine`, `BooksMagazines`, `CandyChips`, `CareHygiene`, `CerealsSpreads`, `CheeseTapas`, `DairyEggs`, `Freezer`, `FruitVegetables`, `HouseholdPet`, `MeatFish`, `PastaRice`, `SaladsMeals`, `SaucesSpices`, `SodaJuices`, `SpecialDiet`, `VegetarianVegan`.
- Min-max normalises and uses `fviz_nbclust` (within-sum-of-squares, k.max=30) to inform the choice of k.
- Runs `kmeans(k=24, nstart=25, algorithm="Lloyd")`.
- Visualises clusters via `fviz_cluster` and per-cluster radar charts (`fmsb::radarchart`) showing department-mix profiles, in groups of 3 clusters per chart (8 charts × 3 = 24 clusters).

The clustering surfaces **behaviour archetypes** (e.g. "drinks-only late-night customer", "weekly bulk-shopper with mixed fresh and packaged"). Anomalous transactions are typically the ones that don't fit any archetype well.

### 6. R: pilot fraud-labelling + logistic regression (`R_scripts/predict 1.Rmd`)

Two pilots were run end-to-end:

- **Pilot 1**: 100 selected customers from `supermarket_3.xlsx`. 10 were manually labelled as fraud (`Class = 1`).
- **Pilot 2**: 100 selected customers from `supermarket_2.xlsx` (66-item slot version). 12 were manually labelled as fraud.

The two pilot frames are stacked (`rbind`) on the `AvgTime → Class` columns, giving a small but balanced-ish dataset. Then:

- 60/40 stratified split via `caTools::sample.split`.
- `glm(Class ~ ., family = binomial())` — logistic regression on every behavioural + department feature.
- ROC curve + AUC via `pROC::roc`.

Logistic regression on engineered features is deliberately the first model — it gives interpretable coefficients that point at *which behaviours* (e.g. unusually low average scan time, an unusually high count of high-margin departments like Beer/Wine or Meat/Fish) the model is using to flag fraud. That interpretability matters more than raw F1 in a retail-loss-prevention context.

## Reproducing

### Python

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn statsmodels termcolor openpyxl
cd scripts/
python main.py     # supervised baselines (needs data/creditcard.csv)
python main1.py    # synthetic anomaly detection sanity check
python main2.py    # Isolation Forest on credit-card data
python main3.py    # Isolation Forest + PCA on supermarket data
```

### R

```r
install.packages(c("dplyr", "splitstackshape", "readxl", "tidyverse",
                   "cluster", "factoextra", "ggplot2", "ranger", "caret",
                   "data.table", "caTools", "pROC", "fmsb", "broom"))
```

Open the `.Rmd` files in RStudio and knit. The scripts read from `~/Downloads/supermarket_2.xlsx` and `~/Downloads/supermarket_3.xlsx` — adjust paths to point at this repo's `data/` folder.

## Notes / known limitations

- The labelled fraud cases in the pilots are hand-labelled by the experimenters; they're a small ground-truth set and the resulting AUC should be read as indicative, not production-grade.
- The R scripts assume specific column-count layouts (74-item slots for Pilot 1, 66 for Pilot 2). Real-world deployment would need a transaction-length-agnostic feature extractor.
- `creditcard.csv` is the public Kaggle benchmark and is *not* a self-checkout dataset; it's used here only as a sanity check that the supervised + unsupervised pipelines behave as expected on extreme class imbalance.
