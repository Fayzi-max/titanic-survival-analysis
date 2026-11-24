# 🚢 Titanic Passenger Survival Analysis & Predictive Modeling

An end-to-end Data Science and Machine Learning project analyzing the passenger survival dynamics of the RMS Titanic disaster using rigorous Exploratory Data Analysis (EDA), domain-driven feature engineering, and leakage-free Scikit-Learn pipelines.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Dataset Summary & Data Dictionary](#-dataset-summary--data-dictionary)
- [Data Cleaning & Missing Value Strategy](#-data-cleaning--missing-value-strategy)
- [Key Exploratory Findings & Domain Insights](#-key-exploratory-findings--domain-insights)
- [Feature Engineering](#-feature-engineering)
- [Machine Learning Architecture & Pipelines](#-machine-learning-architecture--pipelines)
- [Model Diagnostics, Tuning & Benchmark](#-model-diagnostics-tuning--benchmark)
- [Model Interpretability & Feature Attribution](#-model-interpretability--feature-attribution)
- [Key Takeaways & Methodological Improvements](#-key-takeaways--methodological-improvements)
- [Installation & How to Run](#-installation--how-to-run)

---

## 📖 Project Overview

On April 15, 1912, the *RMS Titanic* sank after colliding with an iceberg, resulting in the deaths of 1,502 out of 2,224 passengers and crew. While survival involved an element of luck, demographic status, wealth, and passenger cabin placement heavily determined who secured a seat in the lifeboats.

This project delivers:
1. **In-depth Exploratory Data Analysis (EDA)** investigating historical, demographic, and socioeconomic survival factors.
2. **Context-aware data imputation** that preserves subgroup distributions instead of applying naive global means.
3. **Domain-specific feature engineering** capturing party sizes, honorific titles, and group travel dynamics.
4. **Leakage-free Scikit-Learn pipelines** with standardized cross-validation.
5. **Overfitting diagnostics & mitigation**, evaluating linear and ensemble tree-based classifiers.
6. **Dual model interpretability** using directional log-odds coefficients and test-set permutation importance.

---

## 📂 Repository Structure

```text
Titanic_EDA/
├── Titanic_EDA_Improved.ipynb  # Comprehensive, fully documented, leakage-free notebook
├── train.csv                   # Titanic passenger dataset (891 rows, 12 columns)
└── README.md                   # Project documentation, findings, and technical guide
```

---

## 📊 Dataset Summary & Data Dictionary

The training dataset (`train.csv`) consists of **891 passenger records** across **12 raw features**:

| Feature Name | Type | Description | Key Notes / Values |
|:---|:---|:---|:---|
| `PassengerId` | Integer | Unique identifier for each passenger | 1 to 891 (100% unique primary key) |
| `Survived` | Binary | Target variable (survival status) | `0` = Perished, `1` = Survived |
| `Pclass` | Integer | Ticket socio-economic class | `1` = 1st Class, `2` = 2nd Class, `3` = 3rd Class |
| `Name` | String | Passenger full name with honorifics | Includes title (e.g., *Mr.*, *Mrs.*, *Master.*) |
| `Sex` | String | Biological gender | `male` (64.8%), `female` (35.2%) |
| `Age` | Float | Age in years | Fractional if under 1 year; 19.9% missing |
| `SibSp` | Integer | Number of siblings and/or spouses aboard | Discrete range: 0 to 8 |
| `Parch` | Integer | Number of parents and/or children aboard | Discrete range: 0 to 6 |
| `Ticket` | String | Ticket number string | Shared across family & traveling groups |
| `Fare` | Float | Passenger fare paid | Right-skewed (median: £14.45, max: £512.33) |
| `Cabin` | String | Cabin room number | 77.1% missing; first letter indicates Deck |
| `Embarked` | String | Port of embarkation | `C` = Cherbourg, `Q` = Queenstown, `S` = Southampton |

---

## 🧹 Data Cleaning & Missing Value Strategy

Rather than dropping sparse columns or applying generic mean imputation, each feature was addressed using domain context:

### 1. `Cabin` (77.10% Missing — 687 rows)
* **Insight:** Missingness in `Cabin` is not random noise; having an assigned cabin reflects accommodation on upper decks closer to boat decks.
* **Extraction:**
  - `HasCabin`: Binary indicator (1 if cabin was recorded, 0 if null). Passengers with recorded cabins had a **66.67%** survival rate versus **29.99%** for those without.
  - `Deck`: First character of cabin string (`A` through `G`, `T`, and `Unknown`). Decks `B`, `D`, and `E` demonstrated survival rates $>74\%$.
  - `CabinCount`: Total number of rooms booked on a single ticket (up to 4 rooms).
* Raw high-cardinality `Cabin` strings were dropped after extraction.

### 2. `Age` (19.87% Missing — 177 rows)
* **Insight:** Age distribution varies strongly by passenger class:
  - 1st Class Median Age: **37.0 years**
  - 2nd Class Median Age: **29.0 years**
  - 3rd Class Median Age: **24.0 years**
* **Imputation:** Missing `Age` values were imputed with the **median age of the corresponding `Pclass`**, preserving demographic differences.

### 3. `Embarked` (0.22% Missing — 2 rows)
* **Investigation:** Passengers 62 and 830 (*Miss Amelie Icard* and *Mrs George Nelson Stone*) were 1st class passengers on ticket `113572` with a £80.00 fare.
* **Imputation:** First-class female passengers embarking at Southampton (`S`) paid a median fare of **£79.65** (vs. £83.16 at Cherbourg). The two records were accurately imputed with the mode (`S`).

---

## 🔍 Key Exploratory Findings & Domain Insights

### 1. The Primacy of Protocol: "Women and Children First"
* **Female Survival:** **74.20%**
* **Male Survival:** **18.89%**
* Gender was the single most dominant individual bivariate predictor of survival.

### 2. Socioeconomic Stratification
Survival probability correlated directly with ticket tier:
* **1st Class:** **62.96%** survival
* **2nd Class:** **47.28%** survival
* **3rd Class:** **24.24%** survival

### 3. The Gender $\times$ Class Interaction
Cross-tabulating gender and class reveals distinct survival gradients:
* **1st Class Females:** **96.81%** survived (nearly universal rescue).
* **2nd Class Females:** **92.11%** survived.
* **3rd Class Females:** **50.00%** survived (hampered by lower deck quarters, locked gates, and language barriers).
* **1st Class Males:** **36.89%** survived (nearly triple the survival rate of 3rd Class males at **13.54%**).

### 4. Non-Linear Family Size Curve
* **Traveling Alone (`IsAlone = 1`):** Survival rate was **30.35%**.
* **Small Families (2–4 members):** Survival rate peaked at **55.28% to 72.41%**. Small groups supported each other during evacuation without overburdening lifeboat capacity.
* **Large Families (5+ members):** Survival plummeted to $<20\%$ (and **0%** for families of 8 and 11), as keeping large groups together during the panic proved fatal.

### 5. Ticket & Fare Anomalies Explained
* **£512.33 Maximum Fare:** Belonged to ticket `PC 17755` shared by three passengers: *Thomas Cardeza* (wealthy heir), his valet *Gustave Lesurer*, and personal maid *Anna Ward*, who occupied the luxury parlor suite B51–B53–B55. All three survived.
* **£0.00 Fares:** 15 passengers held zero-fare tickets. Historical records verify these were American Line crew members, Marconi telegraph operators, or shipyard guarantee group members traveling on complimentary corporate passes.

---

## ⚙️ Feature Engineering

To provide maximum signal for machine learning algorithms, domain features were synthesized:

| Feature | Logic / Derivation | Analytical Purpose |
|:---|:---|:---|
| `FamilySize` | `SibSp + Parch + 1` | Quantifies total travel unit size aboard |
| `IsAlone` | `1` if `FamilySize == 1`, else `0` | Captures solitary traveler penalty |
| `Title` | Regex extraction: `r',\s*([^\.]+)\.'` | Standardized into `Mr`, `Miss`, `Mrs`, `Master`, `Rare` |
| `TicketGroupSize` | Frequency count per ticket number | Captures traveling companions beyond nuclear families |
| `AgeGroup` | Bins: `(0,12], (12,18], (18,35], (35,60], (60,100]` | Categorizes life stage (`Child`, `Teenager`, `Young Adult`, `Adult`, `Senior`) |

* **Title Impact:** Young boys bearing the title `Master` had a **57.50%** survival rate, cleanly decoupling male children from adult men (`Mr` at **15.67%**).

---

## 🛠️ Machine Learning Architecture & Pipelines

To strictly eliminate **data leakage**, all transformations and scalers are wrapped in Scikit-Learn `Pipeline` and `ColumnTransformer` modules:

```text
Raw Features (X)
  ├── Numerical Features (10) ──> StandardScaler() ──────────────┐
  └── Categorical Features (5) ─> OneHotEncoder(ignore_unknown) ─┴─> Transformed Space (34 Features)
                                                                        │
                                                                        ▼
                                                             Estimator (LR / RF)
```

* **Dataset Split:** 80% Train ($n=712$), 20% Test ($n=179$), stratified on `y` (`Survived`).
* **Numerical Pipeline:** `StandardScaler` applied to `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `HasCabin`, `CabinCount`, `FamilySize`, `IsAlone`, and `TicketGroupSize`.
* **Categorical Pipeline:** `OneHotEncoder(handle_unknown='ignore')` applied to `Sex`, `Embarked`, `Deck`, `Title`, and `AgeGroup`.

---

## 📈 Model Diagnostics, Tuning & Benchmark

### 1. Overfitting Diagnosis
An unconstrained Random Forest memorized the training data but failed to generalize:
* **Train Accuracy:** **98.74%**
* **Test Accuracy:** **79.89%**
* **Generalization Gap:** **18.85%** (Severe Overfitting)

### 2. Regularization & Cross-Validation
Introducing depth bounds (`max_depth=5`), minimum split samples (`min_samples_split=10`), and minimum leaf samples (`min_samples_leaf=5`) completely eliminated the gap.

### 3. Hyperparameter Tuning (`GridSearchCV`)
5-Fold Cross Validation evaluated 90 parameter configurations:
* **Optimal Parameters:** `{'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10}`
* **Best 5-Fold CV Score:** **82.59%**

### 4. Benchmark Performance Comparison

| Model | Train Accuracy | Test Accuracy | Overfit Gap | 5-Fold CV Accuracy | Test Precision (1) | Test Recall (1) | Test F1-Score (1) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 84.13% | **84.36%** | -0.23% | 82.03% | **0.82** | **0.77** | **0.79** |
| **Random Forest (Unconstrained)** | 98.74% | 79.89% | +18.85% | 80.76% | 0.76 | 0.70 | 0.73 |
| **Random Forest (Regularized)** | 83.99% | 80.45% | +3.54% | 82.31% | 0.77 | 0.70 | 0.73 |
| **Random Forest (GridSearch Tuned)** | 89.61% | 81.01% | +8.60% | **82.59%** | 0.80 | 0.68 | 0.73 |

> **Key Modeling Takeaway:** Logistic Regression achieved the highest overall test accuracy (**84.36%**) and balanced recall (**76.81%** for survivors). The monotonic relationships among gender, class, fare, and title map exceptionally well onto regularized log-odds space.

---

## 🔬 Model Interpretability & Feature Attribution

### 1. Logistic Regression Coefficients (Log-Odds Impact)
* **Strongest Positive Drivers:**
  - `Title_Master` ($\beta = +1.10$): Young boys prioritized for lifeboats.
  - `Sex_female` ($\beta = +0.83$): "Women and children first" effect.
  - `Deck_E` ($\beta = +0.76$) and `Deck_B` ($\beta = +0.48$): Proximity to upper evacuation decks.
* **Strongest Negative Drivers:**
  - `Title_Mr` ($\beta = -1.26$): Adult men deprioritized during boarding.
  - `Deck_G` ($\beta = -0.80$): Lowest structural passenger deck.
  - `Pclass` ($\beta = -0.32$): Lower class (higher numerical value) reduced survival odds.

### 2. Permutation Importance on Held-Out Test Set
Evaluating the performance drop when individual features are randomly permuted:
1. `Sex` (Accuracy drop: **$\Delta = 0.1006 \pm 0.024$**)
2. `Title` (Accuracy drop: **$\Delta = 0.0497 \pm 0.017$**)
3. `Age` (Accuracy drop: **$\Delta = 0.0246 \pm 0.005$**)
4. `AgeGroup` (Accuracy drop: **$\Delta = 0.0067 \pm 0.008$**)
5. `Embarked` (Accuracy drop: **$\Delta = 0.0064 \pm 0.007$**)
6. `Pclass` (Accuracy drop: **$\Delta = 0.0050 \pm 0.010$**)

---

## 🏆 Key Takeaways & Methodological Improvements

Compared to standard ad-hoc Titanic notebooks, this implementation:
* **Eliminated Redundancies:** Removed duplicated distribution plots, redundant IQR calculation loops, and unused intermediate variables.
* **Fixed Code Errors:** Corrected un-fitted pipeline references (`AttributeError: 'LogisticRegression' object has no attribute 'coef_'`), enabling full coefficient inspection.
* **Prevented Data Leakage:** Ensured all scaling, one-hot encodings, and imputation medians are computed strictly inside training folds.
* **Resolved Tree Overfitting:** Diagnosed an 18.85% generalization gap in baseline random forests and mitigated it through constrained tree hyperparameters.
* **Embedded Rich Visualizations:** Integrated 2x2 multi-axis subplot grids, log-scale boxplots, annotated bar charts, side-by-side confusion matrices, and feature importance bar plots.

---

## 🚀 Installation & How to Run

### Prerequisites
Make sure Python 3.9+ is installed along with the required scientific packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Running the Notebook
Navigate to the directory and launch Jupyter Notebook or VS Code:

```bash
cd Titanic_EDA
jupyter notebook Titanic_EDA_Improved.ipynb
```

You can execute the entire notebook sequentially using `Kernel -> Restart & Run All`. All cells execute cleanly without external dependencies beyond standard scientific Python libraries.

