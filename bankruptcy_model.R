
# Bankruptcy Prediction Project
# Using Logistic Regression + Random Forest


# Load libraries
library(ROSE)
library(randomForest)
library(caret)
library(pROC)
library(dplyr)

# Prepare data

train$Bankrupt. <- as.factor(train$Bankrupt.)
test$Bankrupt. <- as.factor(test$Bankrupt.)

# Balance training data using ROSE
set.seed(123)
train_balanced <- ROSE(Bankrupt. ~ ., data = train, N = nrow(train)*2)$data
train_balanced$Bankrupt. <- as.factor(train_balanced$Bankrupt.)


# Train Random Forest
set.seed(123)
rf_model <- randomForest(Bankrupt. ~ ., data = train_balanced, ntree = 500)

#  Predict probabilities on test set
pred_prob_rf <- predict(rf_model, newdata = test, type = "prob")[,2]

# Threshold tuning for rare class
threshold <- 0.1  # Adjust threshold to improve sensitivity
pred_rf <- ifelse(pred_prob_rf > threshold, 1, 0)
pred_rf <- as.factor(pred_rf)

#  Confusion Matrix & Metrics
conf_matrix <- confusionMatrix(pred_rf, test$Bankrupt., positive = "1")
print(conf_matrix)

# ROC Curve & AUC
roc_obj <- roc(test$Bankrupt., pred_prob_rf)
plot(roc_obj, col = "blue", main = "ROC Curve - Random Forest")
auc_value <- auc(roc_obj)
print(paste("AUC:", round(auc_value, 4)))

# Feature Importance
importance_rf <- importance(rf_model)
varImpPlot(rf_model)

# Save ROC
png("outputs/roc_rf.png")
plot(roc_obj, col = "blue", main = "ROC Curve - Random Forest")
dev.off()

# Save feature importance
png("outputs/feature_importance_rf.png")
varImpPlot(rf_model, main = "Feature Importance - Random Forest")
dev.off()


# Save ROC curve
png("outputs/roc_rf.png")
plot(roc_obj, col = "blue", main = "ROC Curve - Random Forest")
dev.off()

# Save Feature Importance
png("outputs/feature_importance_rf.png")
varImpPlot(rf_model, main = "Feature Importance - Random Forest")
dev.off()

# Load data
train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")

# Convert target variable to factor
train$Bankrupt. <- as.factor(train$Bankrupt.)
test$Bankrupt. <- as.factor(test$Bankrupt.)


library(ROSE)
set.seed(123)
train_balanced <- ROSE(Bankrupt. ~ ., data = train, N = nrow(train)*2)$data
train_balanced$Bankrupt. <- as.factor(train_balanced$Bankrupt.)

library(randomForest)
rf_model <- randomForest(Bankrupt. ~ ., data = train_balanced, ntree = 500)

# Predict and evaluate
pred_prob_rf <- predict(rf_model, newdata = test, type = "prob")[,2]
threshold <- 0.1
pred_rf <- ifelse(pred_prob_rf > threshold, 1, 0)
pred_rf <- as.factor(pred_rf)

library(caret)
conf_matrix <- confusionMatrix(pred_rf, test$Bankrupt., positive = "1")
print(conf_matrix)

# Save predictions 
write.csv(data.frame(pred_rf, test$Bankrupt.), "outputs/predictions.csv", row.names = FALSE)

write.csv(train, "data/train.csv", row.names = FALSE)
write.csv(test, "data/test.csv", row.names = FALSE)

source("scripts/bankruptcy_model.R")
