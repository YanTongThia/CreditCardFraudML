# Credit Card Fraud Machine Learning

This machine learning project explores the detection of credit card fraud using a dataset containing over 550,000 records with anonymized features (V1 - V28) and transaction amounts. The binary variable 'Class' is used to label whether a transaction is fraudulent (1) or not (0).

The project begins by developing a logistic regression model to predict the binary 'Class' variable. Variable selection is based on statistical output, and several iterations of the model are created by removing specific variables.

In addition, a Classification and Regression Trees (CART) model is developed, and its complexity is adjusted using the 'cp' parameter. Variable importance is assessed in this context, revealing that transaction amount plays a minimal role in fraud detection.

The analysis concludes by highlighting the trade-off between model accuracy and complexity and emphasizes the importance of achieving high accuracy in fraud detection, even if it requires a more complex model. The comparison between logistic regression and CART models dispels the initial notion that transaction amount is a key indicator of fraud, as machine learning reveals its limited importance in this context.

## Acknowledgements

Data Sources: 
https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023