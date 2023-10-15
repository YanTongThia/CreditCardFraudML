
library(data.table)
options(scipen = 999)

# https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023
# Datset contains over 550,000 records of various anonymized features (V1 - V28) and Transaction Amount labelled with a binary variable indicating whether the transaction is fradulent (1) or not (0).
ccfraud = fread("creditcard_2023.csv")

# Data read into R as Numeric. Convert to Factor.
ccfraud$Class = factor(ccfraud$Class)
levels(ccfraud$Class)

# Develop Logistic Regression Model to Predict Binary Categorical Variable Class 1, 0
log1 = glm(Class ~ .-id,  family = binomial, data = ccfraud)
summary(log1)

# Without expert opinion on variable importance, we will rely on statistic output by model.
# Removing V5 and Amount
log2 = glm(Class ~ .-id -V5 -Amount,  family = binomial, data = ccfraud)
summary(log2)

# Without expert opinion on variable importance, we will rely on statistic output by model.
# Removing V13
log3 = glm(Class ~ .-id -V5 -Amount -V13,  family = binomial, data = ccfraud)
summary(log3)

# P(Y=1) = 1 / (1 + e^-(b0 + b1V1 + b2V2 + .... bNVN))

# Interpretation of Results: 
# For every 1 unit increase in X (Continuous), the odds of Y = 1 multiply by e^bk
# E.g., For every 1 unit increase in V1, increases the odds fraudulent transaction by a factor of 0.50506106
exp(coef(log3))

# Output Probability from Logistic Function for all cases in data
prob = predict(log3, type="response")
# Form Prediction based on Standard Threshold
predict = ifelse(prob > 0.5, 1, 0)

# Confusion Matrix
table(predict,ccfraud$Class, deparse.level = 2)

# Overall Model Accuracy - 0.964935
mean(predict == ccfraud$Class)

library(rpart)
library(rpart.plot)

# Develop CART Model: Growing Tree to Maximum
cart1 <- rpart(Class ~ .-id, data = ccfraud, method = 'class', control = rpart.control(minsplit = 20, cp = 0))

# Variable Importance is determined by a Point-based System
cart1$variable.importance
# Interesting to Note that the Amount Transacted bear little to no importance

# Prune Triggers From Maximum Tree to Minimum Tree
printcp(cart1, digits = 3)

# Determine CV Error Cap based on 1SE Rule
CVerror.cap <- cart1$cptable[which.min(cart1$cptable[,"xerror"]), "xerror"] + cart1$cptable[which.min(cart1$cptable[,"xerror"]), "xstd"]

# Find the optimal CP region whose CV error is just below CVerror.cap in maximal tree cart1.
i <- 1; j<- 4
while (cart1$cptable[i,j] > CVerror.cap) {
  i <- i + 1
}

# Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split.
cp.opt = ifelse(i > 1, sqrt(cart1$cptable[i,1] * cart1$cptable[i-1,1]), 1)

# Prune to Optimal CART Model
cart2 <- prune(cart1, cp = cp.opt)

printcp(cart2, digits = 3)
print(cart2) # 634 splits

rpart.plot(cart2, nn = T) # Too many Nodes to Plot
summary(cart2)
cart2$variable.importance

# CART 2 has very high complexity due to low complexity cost issued.
# If we were to artificially increase the complexity parameter, to charge a higher cost for higher complexity, 
# CART model will be drastically simplified, increasing understanding of model outcomes.
cart3 <- prune(cart1, cp = 0.001) # 24 splits
printcp(cart3, digits = 3)
print(cart3)
summary(cart3)
rpart.plot(cart3) # Able to Plot and Visualize Flow

# Prediction based on Trainset on Cart 3
predicted.value = predict(cart3, type="class")

# Confusion Matrix
table(predicted.value,ccfraud$Class, deparse.level = 2)

# Overall Model CART 3 Accuracy 0.9646642
mean(predicted.value == ccfraud$Class)

# Prediction based on Trainset on CART 2 (Optimal Model 1SE Rule)
predicted.value = predict(cart2, type="class")

# Confusion Matrix
table(predicted.value,ccfraud$Class, deparse.level = 2)

# Overall Model Accuracy 0.998579
mean(predicted.value == ccfraud$Class)

# Increasing the Complexity Parameter (cp) is observed to result in a reduction in Model Accuracy.

#Model developers must carefully balance Model Accuracy and Model Complexity to find an optimal trade-off, ensuring the model is sufficiently effective for its intended purpose.

# Given the gravity of fraud-related situations and the need for accurate detection, a higher level of accuracy is often a desirable goal to effectively identify instances of fraud.

# To determine the optimal tree sequence between the maximum and minimum number of trees, the cp parameter can be further fine-tuned.

# The CART Model serves as a precursor to modern machine learning models like Random Forest, which defaults to using 500 CART Trees without encountering issues of overfitting.

# It is worth noting that, when comparing the two models, Logistic Regression and CART, the transaction amount does not play a significant role in detecting fraud. This contrasts with the initial notion that larger monetary transactions are more suspicious and warrant investigation. Through Machine learning, however, it is revealed that the transaction amount actually holds little to no importance in the context of fraud detection.



