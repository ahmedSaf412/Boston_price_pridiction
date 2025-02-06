# Boston Housing Dataset Analysis

## Overview
This project analyzes the Boston Housing Dataset, a well-known dataset collected by the U.S. Census Service concerning housing in the Boston Massachusetts area. The analysis includes data exploration, feature engineering, and various regression models to predict housing prices.

## Dataset Description
The dataset contains information about 506 census tracts with the following features:

### Target Variable
- **MEDV**: Median value of owner-occupied homes in $1,000s (capped at $50,000)

### Features
- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxides concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to five Boston employment centers
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: 1000(Bk - 0.63)² where Bk is the proportion of Black residents by town
- **LSTAT**: % lower status of the population

## Key Insights

### Data Quality
- Several features show significant outliers:
  - CRIM: 13.04% outliers
  - ZN: 13.44% outliers
  - B: 15.22% outliers
- MEDV is censored at $50,000, affecting approximately 8% of the data

### Feature Relationships
- Strong predictors of housing prices (MEDV):
  - LSTAT (negative correlation)
  - RM (positive correlation)
  - PTRATIO (negative correlation)
  - NOX (negative correlation)
- High multicollinearity between TAX and RAD features

### Model Performance
The following regression models were evaluated:

1. KNeighbors Regressor: MSE 0.024252
2. Ridge Regression: MSE 0.030029
3. Decision Tree Regressor: MSE 0.030042
4. Linear Regression: MSE 0.030216
5. SVR (RBF Kernel): MSE 0.031215
6. Lasso Regression: MSE 0.045792
7. Polynomial Regression (degree=3): MSE 3421.874936

## Requirements
```
python>=3.7
pandas
numpy
scikit-learn
seaborn
matplotlib
scipy
```



2. Install required packages
```bash
pip install -r requirements.txt
```

## Usage
```python
from housing_analysis import analyze_boston_housing

# Load and analyze data
results = analyze_boston_housing(data)

# Access analysis results
print(results['key_insights'])
print(results['recommendations'])
```

## Recommendations
1. Data Preprocessing:
   - Remove or treat censored MEDV values for more accurate modeling
   - Apply non-linear transformations to CRIM, ZN, and B features
   - Implement robust scaling to handle outliers

2. Feature Engineering:
   - Consider feature selection to handle multicollinearity
   - Create interaction terms for related features
   - Investigate polynomial features for non-linear relationships

3. Modeling:
   - Experiment with ensemble methods
   - Use cross-validation for more robust model evaluation
   - Consider non-linear models due to relationships in the data

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Data source: U.S. Census Service
- Original dataset creators: Harrison, D. and Rubinfeld, D.L.
- Dataset hosted by: UCI Machine Learning Repository
