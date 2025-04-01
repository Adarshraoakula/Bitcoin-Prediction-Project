# Bitcoin Price Prediction Project

This project aims to predict Bitcoin price trends by analyzing the sentiment of various data sources, including news articles, Wikipedia edits, and historical Bitcoin price data. The project employs a combination of machine learning techniques such as sentiment analysis, time series forecasting, and backtracking to ensure prediction accuracy. The goal is to build a robust model that forecasts Bitcoin price fluctuations using sentiment-driven insights.

---

## 📚 Table of Contents

- [Project Description](#project-description)
- [Datasets](#datasets)
- [Methodology](#methodology)
  - [Sentiment Extraction](#sentiment-extraction)
  - [Price Prediction](#price-prediction)
  - [Backtracking and Evaluation](#backtracking-and-evaluation)
- [Implementation](#implementation)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

---

## 📌 Project Description

Bitcoin is a highly volatile digital asset, and predicting its price fluctuations has been a significant challenge for analysts, investors, and traders alike. This project leverages sentiment data derived from multiple sources—news articles, Wikipedia edits, and historical Bitcoin data—to predict future price trends of Bitcoin. By analyzing the sentiment around Bitcoin and combining it with historical price data, this project builds a predictive model aimed at capturing price movements in a timely and accurate manner.

### 🔑 Key Components:

#### Sentiment Analysis
- **VADER**: Lexicon and rule-based sentiment analysis tool suitable for short texts like headlines.
- **BERT**: Deep learning model that captures context from both directions (left-to-right and right-to-left) for full-text sentiment analysis.
- **FinBERT (Optional)**: A BERT-based model fine-tuned for financial texts, used for sentiment analysis of financial news.

#### Price Prediction
- Combines historical Bitcoin price data with sentiment scores.
- Utilizes models such as Linear Regression and LSTM to forecast price trends.

#### Backtracking & Model Evaluation
- Refines predictions using backtracking to improve accuracy iteratively.
- Evaluated with metrics like MAE, RMSE, and R².

---

## 📂 Datasets

The following datasets are used in the project:

- **Bitcoin Historical Data**: 
  - Sourced from Yahoo Finance using the `yfinance` API (Open, High, Low, Close, Volume).
- **Wikipedia Edits**: 
  - Scraped to analyze public perception changes over time.
- **News Articles**: 
  - Collected via APIs like SerpAPI to perform sentiment analysis.

---

## 🧠 Methodology

### Sentiment Extraction

#### 🔹 VADER Sentiment Analysis
- **Purpose**: Rapid estimation for short text (e.g., headlines).
- **Application**: Analyzes news snippets and article headlines.

#### 🔹 BERT
- **Purpose**: Extracts deeper context-based sentiment.
- **Application**: Used on full news articles and Wikipedia content.

#### 🔹 FinBERT (Optional)
- **Purpose**: Tailored for financial text sentiment extraction.
- **Application**: Provides sentiment scores for financial news content.

---

### Price Prediction

- Uses historical Bitcoin price data (OHLCV).
- Incorporates sentiment scores as additional features.
- Applies models such as:
  - Linear Regression
  - Random Forest
  - LSTM (Long Short-Term Memory networks)

---

### Backtracking and Evaluation

- **Backtracking**:
  - Iterative prediction refinement using new data updates.
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- **Cross-validation** ensures model generalizability.

---

## 🚀 Implementation

### 1. Data Collection
- Historical price data from Yahoo Finance via `yfinance`.
- News articles and Wikipedia edits collected via APIs like SerpAPI.

### 2. Data Preprocessing
- Clean and merge sentiment data with historical prices.

### 3. Model Training
- Trains various models using both price data and sentiment scores.

### 4. Prediction
- Generates price forecasts using trained models.

---

## 🧮 Modeling

The following ML models are explored:

- **Linear Regression**: Baseline model.
- **Random Forest**: Handles complex feature relationships.
- **LSTM**: Captures sequential dependencies in time series data.

---

## 📊 Evaluation

Evaluation is done using:

- 📌 **Mean Absolute Error (MAE)**
- 📌 **Root Mean Squared Error (RMSE)**
- 📌 **R² Score**

Models are fine-tuned using:
- Cross-validation
- Hyperparameter optimization

---

## ⚙️ Getting Started

### ✅ Prerequisites

Make sure you have Python 3.x installed. Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn transformers yfinance vaderSentiment
