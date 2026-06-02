# Titanic Survival Prediction

A machine learning project exploring what passenger characteristics most influenced survival outcomes aboard the RMS Titanic. Using demographic and ticketing data, this project builds and compares classification models to predict whether a given passenger survived the disaster.

---

## The Question

> *"What sorts of people were more likely to survive?"*

On April 15, 1912, the Titanic sank with 1,502 of 2,224 passengers and crew aboard. While survival was partly a matter of luck, patterns in the data reveal that factors like gender, passenger class, and family size played a significant role. This project aims to quantify those patterns through predictive modeling.

---

## Dataset

The dataset contains **891 passenger records** across 12 features:

| Feature | Description |
|---|---|
| `Survived` | Target variable — 0 = No, 1 = Yes |
| `Pclass` | Passenger class (1st, 2nd, 3rd) |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Fare` | Ticket fare |
| `Cabin` | Cabin number |
| `Embarked` | Port of embarkation (C, Q, S) |

---

## Methodology

### Data Cleaning

Missing values were addressed before modeling:

- **Age** — Imputed with the median to preserve distribution shape
- **Cabin** — High missingness (~77%) led to a categorical fill of `'Unknown'`
- **Embarked** — Two missing rows filled with the mode

### Feature Engineering

New features were derived to capture additional signal:

- **`FamilySize`** — `SibSp + Parch + 1`, representing total family unit aboard
- **`IsAlone`** — Binary flag for solo travelers (`FamilySize == 1`)
- **`Title`** — Extracted from passenger names (e.g., *Mr., Mrs., Miss, Master*); rare titles were grouped, and French variants (`Mlle`, `Mme`, `Ms`) were normalized
- **`Deck`** — First letter of cabin number; missing values coded as `'Unknown'`
- **`TicketGroup`** — Count of passengers sharing the same ticket number

Raw identifiers (`PassengerId`, `Name`, `Ticket`, `Cabin`) were dropped after extraction.

### Encoding

Categorical features (`Sex`, `Embarked`, `Title`, `Deck`) were one-hot encoded using `pd.get_dummies`.

---

## Exploratory Data Analysis

Key findings from the EDA:

- **Gender** was the strongest survival predictor — females survived at a significantly higher rate than males
- **Passenger class** correlated sharply with survival; 1st-class passengers had the best odds
- **Age** skewed toward young and middle-aged adults; children showed slightly elevated survival rates
- **Fare** was right-skewed with notable outliers, and negatively correlated with `Pclass`

---

## Modeling

Data was split 80/20 for training and evaluation. Four models were trained and compared:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.8156 | 0.7808 | 0.7703 | 0.7755 |
| Decision Tree (Unconstrained) | — | — | — | — |
| Decision Tree (Pruned, `max_depth=4`) | **0.8268** | **0.8116** | 0.7568 | **0.7832** |
| Decision Tree (GridSearchCV Optimized) | 0.8212 | 0.8088 | 0.7432 | 0.7746 |

The unconstrained tree overfit to the training data and was excluded from the final comparison. Pruning and hyperparameter tuning both improved generalization, with the pruned tree achieving the best overall performance on the test set.

---

## Results & Takeaways

The pruned Decision Tree (82.7% accuracy, 0.78 F1) offered the best balance of precision and recall. More broadly, the analysis confirmed that **gender, passenger class, and fare** were the most influential survival factors — consistent with the historical record of women and higher-class passengers being prioritized for lifeboats.

---

## Tech Stack

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`
