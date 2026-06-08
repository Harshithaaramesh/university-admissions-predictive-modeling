# Logistic Regression for ADMISSION
library(caret)
myData <- read.csv("clean_university_data.csv")
set.seed(1)

myIndex <- createDataPartition(myData$Admitted, p = 0.70, list = FALSE)
TData <- myData[myIndex, ]
VData <- myData[-myIndex, ]

# 1) Fit logistic regression model
Model1 <- glm(
  Admitted ~ HSGPA + SAT_ACT + Sex + White + Asian + Edu_Parent1 + Edu_Parent2,
  family = binomial,
  data   = TData
)

summary(Model1)   

# 2) Predict on validation data
pHat1 <- predict(Model1, VData, type = "response")   # predicted probabilities
yHat1 <- ifelse(pHat1 >= 0.5, 1, 0)                  # convert to 0/1 class

# 3) Compute Accuracy 
acc1 <- mean(VData$Admitted == yHat1)
sprintf("Admission model accuracy = %.2f%%", 100 * acc1)

# ADMISSION – CLASSIFICATION TREE
library(caret)
library(rpart)
library(rpart.plot)
library(pROC)
library(gains)

set.seed(1)

# 1) Load cleaned dataset
myData <- read.csv("clean_university_data.csv", stringsAsFactors = FALSE)

# Make Admitted a factor (0/1 as levels)
myData$Admitted <- as.factor(myData$Admitted)

# 2) Train/Validation split (70/30) – stratified on Admitted
myIndex <- createDataPartition(myData$Admitted, p = 0.7, list = FALSE)
trainSet <- myData[myIndex, ]
validationSet <- myData[-myIndex, ]

# 3) Default tree
default_tree <- rpart(Admitted ~ ., data = trainSet, method = "class")
prp(default_tree, type = 1, extra = 1, under = TRUE, main = "Default Tree – Admitted")

# 4) Full tree (cp = 0, fully grown)
full_tree <- rpart(
  Admitted ~ .,
  data = trainSet,
  method = "class",
  cp = 0,
  minsplit = 2,
  minbucket = 1
)
prp(full_tree, type = 1, extra = 1, under = TRUE, main = "Full Tree – Admitted")

# 5) Complexity parameter (cp) table
printcp(full_tree)

# Choose cp with lowest xerror (best pruned size)
best_cp <- full_tree$cptable[which.min(full_tree$cptable[, "xerror"]), "CP"]
cat("Best cp for Admitted tree:", best_cp, "\n")

# 6) Pruned tree (final model for Admitted)
pruned_tree <- prune(full_tree, cp = best_cp)
prp(pruned_tree, type = 1, extra = 1, under = TRUE, main = "Pruned Tree – Admitted")

# 7) Performance on validation set
pred_class <- predict(pruned_tree, validationSet, type = "class")
cm_admit <- confusionMatrix(pred_class, validationSet$Admitted, positive = "1")
cm_admit

# Gains table
validationSet$Admitted <- as.numeric(as.character(validationSet$Admitted))
pred_prob_val <- predict(pruned_tree, validationSet, type = "prob")
gains(validationSet$Admitted, pred_prob_val[, 2])


# 8) SCORE THE ENTIRE CLEAN DATASET WITH ADMIT PROB
# Reload full cleaned data 
fullData <- read.csv("clean_university_data.csv", stringsAsFactors = FALSE)
fullData$Admitted <- as.factor(fullData$Admitted)

# Predicted class and probabilities for EVERY record
full_pred_class <- predict(pruned_tree, fullData, type = "class")
full_pred_prob  <- predict(pruned_tree, fullData, type = "prob")[, 2]   # prob of class "1" (Admitted)

# Append to dataset
fullData$Admit_Tree_Pred <- full_pred_class
fullData$Admit_Tree_Prob <- full_pred_prob

# Save as new file – this will be the input to Question 2 (Enrollment)
write.csv(fullData, "clean_university_with_AdmitProb.csv", row.names = FALSE)

cat("Saved scored dataset with admission probabilities as: clean_university_with_AdmitProb.csv\n")

