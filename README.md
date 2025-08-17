# ShopSaver AI - Customer Discount Abuse Detector & Spend Predictor

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-lightgrey?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![XGBoost](https://img.shields.io/badge/XGBoost-005FEA?style=flat&logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![Plotly](https://img.shields.io/badge/Plotly-231F20?style=flat&logo=plotly&logoColor=white)](https://plotly.com/python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Live Demo](#live-demo)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Data Requirements & Column Mapping](#data-requirements--column-mapping)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Model Performance](#model-performance)
- [Business Insights & Recommendations](#business-insights--recommendations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

ShopSaver AI is an interactive Streamlit application designed to help e-commerce businesses optimize their discount strategies and forecast customer spending. By analyzing customer transaction data, it identifies potential discount abuse patterns and predicts future monthly spending, providing actionable insights to enhance profitability and customer relationship management. Extract the demo .csv file from Online Retail.zip for the demo dataset.

## Features

* **Customer Discount Abuse Detection:** Identify customers who exhibit patterns indicative of discount code abuse (e.g., primarily purchasing with discounts, high frequency of discount use).
* **Customer Spend Prediction:** Forecast estimated monthly spending for individual customers to aid in inventory planning, targeted marketing, and revenue forecasting.
* **Advanced Feature Engineering:** Utilizes a rich set of behavioral features, including RFM (Recency, Frequency, Monetary) metrics, temporal patterns (peak purchase hour/day), and category diversity (entropy).
* **Enhanced Machine Learning Models:** Employs robust XGBoost (for classification) and Gradient Boosting Regressor (for regression) models with refined hyperparameters for potentially improved accuracy.
* **Interactive Streamlit Dashboard:** A user-friendly web interface for data upload, flexible column mapping, feature generation, model training, and performance visualization.
* **Dynamic Column Mapping:** Allows users to map columns from **any** transactional CSV file to the model's expected fields, making the tool highly adaptable.


## Live Demo

Experience ShopSaver AI live! You can access the deployed application here:

[**Launch ShopSaver AI Application**](https://shopsaverai.streamlit.app)

## How It Works

The application follows a standard machine learning pipeline:

1.  **Data Ingestion:** Users upload their e-commerce transaction data in CSV format.
2.  **Column Mapping:** Users specify which columns in their CSV correspond to the critical fields the model needs.
3.  **Data Preprocessing:** The raw data is cleaned, validated, and transformed.
4.  **Feature Engineering:** A comprehensive set of advanced features is extracted for each unique customer, capturing their purchasing habits, frequency, monetary value, temporal preferences, and product diversity.
5.  **Model Training:** Two distinct machine learning models are trained:
    * An XGBoost Classifier for discount abuse detection.
    * A Gradient Boosting Regressor for customer spend prediction.
6.  **Prediction & Insights:** The trained models predict the likelihood of discount abuse and estimated spend. The dashboard provides aggregated business insights and individual user analysis.

## Getting Started

### Prerequisites

* Python 3.9+
* `pip` (Python package installer)

### Installation

1.  **Clone the repository (or download `app.py`):**
    ```bash
    git clone [https://github.com/Kathitjoshi/ShopSaver_AI.git](https://github.com/Kathitjoshi/ShopSaver_AI.git)
    cd YOUR_REPO_NAME
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install the required packages:**
    ```bash
    pip install pandas numpy streamlit plotly scikit-learn xgboost shap scipy
    ```

### Running the Application

1.  **Ensure your virtual environment is active.**
2.  **Run the Streamlit app from your terminal:**
    ```bash
    streamlit run app.py
    ```
    This command will open the application in your default web browser.

## Data Requirements & Column Mapping

ShopSaver AI is designed to work with **any transactional CSV file**. To ensure the model functions correctly, you must map the columns from your CSV to the model's expected fields on the "Data Analysis" page.

**Core Required Columns for Mapping (5 Fields):**
These fields are essential for the model's core functionality, especially for calculating spend-related metrics.

* **User ID Column:** Unique identifier for each customer (e.g., `CustomerID`).
* **Event Time Column:** Timestamp of each transaction (e.g., `InvoiceDate`). Must be parseable as a datetime object by pandas.
* **Product ID Column:** Unique identifier for each product (e.g., `StockCode`).
* **Unit Price Column:** The price of a single unit of the product (e.g., `UnitPrice`).
* **Quantity Column:** The quantity of items purchased in that transaction line (e.g., `Quantity`). **This is critical for accurate spend calculation.**

**Optional Columns for Mapping (Recommended for richer analysis):**

* **Invoice/Session ID Column:** Identifier for a complete transaction or session (e.g., `InvoiceNo`). Used to group events into sessions. If not provided, a fallback session ID is generated.
* **Description/Category Column:** A textual description or category of the product (e.g., `Description`). Used for category diversity and entropy calculation. If not provided, 'unknown\_category' is used.
* **Discount Applied Column:** A column explicitly stating if a discount was applied to the transaction (e.g., `IsDiscounted`, `DiscountValue`, `True`/`False`, `1`/`0`). If left blank, a heuristic based on `UnitPrice` variance will be used for discount detection, which may be less accurate.
* **Country Column:** The country of the customer or transaction (e.g., `Country`). Used for geographical features. If not provided, `num_unique_countries` defaults to 1.

## Feature Engineering

The `engineer_features` method calculates a wide array of descriptive and predictive features for each unique user, providing a comprehensive understanding of their behavior:

* **RFM (Recency, Frequency, Monetary) Metrics:** `recency_days`, `purchases` (frequency), `total_spend` (monetary).
* **Activity & Engagement:** `total_events`, `events_per_day`, `unique_sessions`, `avg_events_per_session`, `customer_lifetime_days`, `avg_time_between_purchases`.
* **Purchase Composition:** `unique_products`, `unique_categories`, `avg_order_value`, `avg_quantity_per_purchase`, `avg_price_per_item`, `total_quantity_purchased`.
* **Discount Behavior:** `discount_dependency`.
* **Price Dynamics:** `price_std`, `min_price`, `max_price`.
* **Conversion Rates (context-dependent):** `overall_conversion_rate`, `view_to_cart_rate`, `cart_to_purchase_rate`. (Note: `views` and `carts` features might be zero for purely transactional datasets).
* **Diversity Measures:** `category_diversity`, `category_entropy`, `num_unique_brands`, `num_unique_countries`.
* **Temporal Patterns:** `most_common_purchase_hour`, `most_common_purchase_day_of_week`.

## Machine Learning Models

* **Discount Abuse Detection:** Uses `XGBoostClassifier` (from `xgboost` library). A highly efficient and powerful gradient boosting algorithm for classification.
* **Spend Prediction:** Uses `GradientBoostingRegressor` (from `sklearn.ensemble`). Another robust boosting algorithm, well-suited for regression tasks.

Both models undergo an initial round of hyperparameter tuning with increased `n_estimators` and `max_depth` to enhance their predictive capabilities.

## Model Performance

Upon training, the application displays key performance metrics:

* **Classification Accuracy:** Measures the overall correctness of the discount abuse predictions.
* **Regression R² Score:** Indicates how well the spend prediction model explains the variability in the target variable. A value closer to 1.0 is better. A negative R² indicates the model performs worse than simply predicting the mean.
* **Regression Mean Squared Error (MSE):** Measures the average squared difference between predicted and actual spend values. Lower values indicate better accuracy.

## Business Insights & Recommendations

The "Business Dashboard" and "User Prediction" pages provide actionable insights:

* **Identify High-Risk Users:** Flag customers with a high probability of discount abuse for targeted interventions and fraud prevention.
* **Optimize Discount Strategies:** Understand the dependency of customers on discounts to refine promotional campaigns and improve profitability.
* **Personalize Marketing:** Tailor promotions, product recommendations, and loyalty programs based on predicted spend and behavioral patterns.
* **Customer Retention:** Identify customers with declining engagement or predicted spend to proactively re-engage them.

## Future Enhancements

* **Robust Temporal Validation for Spend Prediction:** Implement a strict time-based train-test split (e.g., train on data up to Month X, predict for Month X+1) for more realistic future spend forecasting.
* **Advanced Discount Feature Engineering:** Incorporate features like average discount value, total discount received, and analysis of specific discount types.
* **More Sophisticated Anomaly Detection:** Explore unsupervised learning methods (e.g., clustering, isolation forests) for identifying unusual spending or discount usage patterns.
* **Interactive SHAP Plots:** Integrate interactive SHAP force plots on the "User Prediction" page to provide even deeper, interactive model interpretability.
* **Model Deployment Strategies:** Investigate deploying the trained models as a backend service for real-time inference in production systems.
* **Automated Hyperparameter Optimization:** Integrate libraries like `Optuna` or `GridSearchCV`/`RandomizedSearchCV` for systematic hyperparameter tuning.
* **Categorical Feature Encoding:** Implement more advanced encoding techniques for categorical features like `product_id` or `category_code`.

## Contributing

Contributions are welcome! Please feel free to open issues, submit pull requests, or suggest improvements.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

* **Your Name/Handle:** [Kathitjoshi]

