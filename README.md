# 🛍️ Zara Sales Analysis - Machine Learning Project

A comprehensive machine learning project analyzing Zara sales data with exploratory data analysis, feature engineering, PCA, and predictive modeling.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📊 Project Overview

This project demonstrates a complete end-to-end machine learning pipeline for sales prediction, including:

- Data cleaning and preprocessing
- Comprehensive exploratory data analysis (EDA)
- Feature engineering
- Principal Component Analysis (PCA) for dimensionality reduction
- Multiple ML models training and comparison
- Model evaluation and insights

## 🚀 Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Run the complete analysis
python complete_zara_analysis.py
```

## 📁 Dataset

- **Source**: Zara Sales Data
- **Size**: 20,254 products
- **Features**: 17 columns including product attributes, pricing, positioning, and seasonality
- **Target Variable**: Sales Volume

### Key Features:
- Product Position (Aisle, End-cap, Front of Store)
- Promotion (Yes/No)
- Price (USD)
- Season (Winter, Summer, Autumn, Spring)
- Section (MAN, WOMAN)
- Material, Origin, and more

## 🔍 Analysis Components

### 1. Data Cleaning
- Missing value detection and imputation
- Duplicate removal
- Data type validation

### 2. Exploratory Data Analysis (EDA)

**Univariate Analysis:**
- Distribution of sales volume and prices
- Categorical feature analysis

**Bivariate Analysis:**
- Correlation analysis
- Sales vs Price relationships
- Sales by promotion, position, season, and section

### 3. Feature Engineering
Created new features:
- `price_category`: Budget, Mid-Range, Premium, Luxury
- `sales_category`: Low, Medium, High, Very High
- Seasonal indicators (binary features)
- `price_sales_ratio`: Efficiency metric

### 4. Principal Component Analysis (PCA)
- Variance explained analysis
- Dimensionality reduction (95% variance retained)
- Component loadings visualization
- 2D projection of data

### 5. Machine Learning Models

| Model | Type | Features |
|-------|------|----------|
| Linear Regression | Baseline | Original |
| Ridge Regression | Regularized | Original |
| Random Forest | Ensemble | Original |
| Gradient Boosting | Ensemble | Original |
| Random Forest | Ensemble | PCA |
| Gradient Boosting | Ensemble | PCA |

**Evaluation Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score (Coefficient of Determination)

## 📈 Visualizations

The project generates 13 high-quality visualizations:

1. Univariate numerical distributions
2. Univariate categorical analysis
3. Correlation matrix heatmap
4. Price vs Sales scatter plots
5. Sales by categorical features
6. Feature engineering results
7. PCA variance explained (scree plot)
8. PCA 2D visualization
9. PCA component loadings
10. Model performance comparison
11. Actual vs Predicted plots
12. Residual analysis
13. Feature importance ranking

## 📊 Results

The analysis provides:
- ✅ Best performing model identification
- ✅ Feature importance rankings
- ✅ Impact of promotions on sales
- ✅ Seasonal sales patterns
- ✅ Price optimization insights
- ✅ PCA dimensionality reduction benefits

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **matplotlib & seaborn** - Visualization
- **scikit-learn** - Machine learning algorithms and PCA

## 📝 Usage

### Option 1: Python Script
```bash
python complete_zara_analysis.py
```

### Option 2: Jupyter Notebook
```bash
jupyter notebook Zara_Sales_Complete_Analysis.ipynb
```

## 📂 Project Structure

```
├── Zara_sales_EDA.csv                    # Dataset
├── complete_zara_analysis.py             # Main analysis script
├── Zara_Sales_Complete_Analysis.ipynb    # Jupyter notebook
├── requirements.txt                      # Python dependencies
├── HOW_TO_RUN.txt                        # Quick start guide
├── PROJECT_SUMMARY.txt                   # Detailed documentation
└── README.md                             # This file
```

## 🎯 Key Insights

- Products with promotions show significantly different sales patterns
- Product position (Aisle, End-cap, Front) impacts sales volume
- Seasonal variations affect sales performance
- PCA reduces features by ~50% while maintaining 95% variance
- Ensemble methods (Random Forest, Gradient Boosting) perform best

## ⏱️ Runtime

Expected runtime: **5-10 minutes** (depending on hardware)

## 📋 Requirements

See `requirements.txt` for full list. Main dependencies:
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Senior ML Developer**
- Project Date: November 2025

## 🙏 Acknowledgments

- Dataset: Zara Sales Data
- Libraries: scikit-learn, pandas, matplotlib, seaborn

---

⭐ If you find this project useful, please consider giving it a star!
