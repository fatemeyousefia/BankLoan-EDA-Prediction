# Bank Personal Loan | EDA & Prediction

Exploratory data analysis and classification models to predict which bank customers are more likely to accept a personal loan offer.

> Completed as part of the **Data Science & ML Course (IMT)** by Mohamadreza Momeni.

## Business Problem

A bank wants to target its next personal loan campaign more efficiently. In the previous campaign, only about **9.6%** of customers accepted the offer, so contacting every customer is inefficient.

This project explores customer and banking data and builds classification models to estimate the probability that a customer will accept a personal loan offer.

Because the target variable is highly imbalanced, **accuracy alone can be misleading**. The models are therefore evaluated using precision, recall, F1-score, ROC-AUC, PR-AUC, and log loss.

## Dataset

The dataset contains **5,000 bank customers** and information about demographics, income, spending, mortgage, and banking products.

### Target

* `Personal Loan` — 1 = accepted, 0 = rejected

### Main Features

| Feature            | Description                                               |
| ------------------ | --------------------------------------------------------- |
| Age                | Customer's age                                            |
| Experience         | Years of professional experience                          |
| Income             | Annual income ($000)                                      |
| ZIP Code           | Customer's residential area                               |
| Family             | Number of family members                                  |
| CCAvg              | Average monthly credit card spending ($000)               |
| Education          | 1: Undergrad, 2: Graduate, 3: Advanced/Professional       |
| Mortgage           | Value of house mortgage ($000)                            |
| Securities Account | Whether the customer has a securities account             |
| CD Account         | Whether the customer has a certificate of deposit account |
| Online             | Whether the customer uses online banking                  |
| CreditCard         | Whether the customer has a bank-issued credit card        |

**Source:** Bank Personal Loan Modelling dataset.

## Approach

### 1. Data Cleaning

* Negative `Experience` values were treated as sign errors and converted to absolute values.
* `CCAvg` values were converted to numeric values and annualized by multiplying by 12 so that they are expressed on the same annual basis as `Income`.
* One invalid 4-digit ZIP code was removed.
* ZIP codes were grouped by their first three digits.
* Rare ZIP3 groups with fewer than 50 customers were combined into `Other`.

### 2. Exploratory Data Analysis

The analysis includes:

* Distribution analysis
* Box plots
* Category-level loan acceptance rates
* Relationships between customer characteristics and loan acceptance
* Correlation analysis

Income shows the strongest relationship with loan acceptance in the exploratory analysis. In this dataset, customers with an income below 60 did not accept the loan offer.

Age and Experience are also highly correlated, so their individual coefficients are not interpreted separately.

### 3. Modeling

* Stratified 75/25 train-test split
* Preprocessing implemented inside a scikit-learn `Pipeline`
* Min-max scaling for numerical features
* One-hot encoding for categorical features
* Logistic Regression
* Class-weighted Logistic Regression
* Gaussian Naive Bayes
* K-Nearest Neighbors
* Cross-validation for model and hyperparameter selection
* Decision-threshold analysis
* Final evaluation on the untouched test set

Keeping preprocessing inside the pipeline ensures that information from the test set is not used during model fitting.

## Model Evaluation

Because the positive class represents only a small proportion of customers, the project focuses on metrics that provide more information than accuracy alone:

* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Log loss

The decision threshold is also analyzed because changing the threshold can alter the trade-off between identifying more potential loan acceptors and reducing false positives.

## Key Findings

* The dataset is highly imbalanced, with approximately **9.6%** of customers accepting the personal loan offer.
* Income shows a strong relationship with loan acceptance.
* Other customer and banking characteristics also contribute to differences in acceptance rates.
* Different models produce different precision-recall trade-offs.
* Decision-threshold selection can substantially change the balance between precision and recall.
* Accuracy alone is not an appropriate metric for evaluating this problem.

## Limitations

* The analysis uses a single stratified train-test split.
* Repeated or nested cross-validation could provide a more stable estimate of generalization performance.
* The dataset comes from a specific marketing campaign and should not be treated as a general lending or credit-risk model.
* The project focuses on predicting campaign response rather than making actual lending or underwriting decisions.

## Future Work

Potential next steps include:

* Testing tree-based models such as Decision Tree, Random Forest, and Gradient Boosting
* Further tuning of Logistic Regression
* Evaluating model calibration
* Exploring more robust validation strategies
* Comparing models using business cost-sensitive metrics based on campaign costs

## Project Structure

```text
BankLoan-EDA-Prediction/
├── notebook/
│   └── BankLoan-EDA-Prediction.ipynb
├── requirements.txt
└── README.md
```


The dataset is not included in the repository. To reproduce the analysis, place the dataset in the working directory as:

```text
data/Bank_Personal_Loan_Modelling.csv
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/fatemeyousefia/BankLoan-EDA-Prediction.git
cd BankLoan-EDA-Prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Place the dataset at:

```text
data/Bank_Personal_Loan_Modelling.csv
```

Then open the notebook in Jupyter and run all cells.

> Note: Some interactive Plotly visualizations may not display fully in GitHub's notebook viewer. Open the notebook locally in Jupyter to view the interactive charts.

## Tools

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Scikit-learn
* Jupyter Notebook

## Author

**Fatemeh Yousefi Amiri**

[GitHub](https://github.com/fatemeyousefia) · [LinkedIn](https://www.linkedin.com/in/fatemeyousefi)

