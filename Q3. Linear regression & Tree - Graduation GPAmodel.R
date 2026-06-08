# Q3: Linear Regression – College GPA
options(scipen = 999)

# 1. Read cleaned dataset
myData <- read.csv("clean_university_data.csv")

# Keep only rows where College_GPA is not missing
myData_reg <- subset(myData, !is.na(College_GPA))

# 2. Build three linear models
# Model 1: all main predictors
Model_1 <- lm(College_GPA ~ HSGPA + SAT_ACT + Admitted + Enrolled, data = myData_reg)

# Model 2: drop SAT_ACT
Model_2 <- lm(College_GPA ~ HSGPA + Admitted + Enrolled, data = myData_reg)

# Model 3: drop HSGPA
Model_3 <- lm(College_GPA ~ SAT_ACT + Admitted + Enrolled, data = myData_reg)

# 3. Summaries (R², Adj R², p-values)
sum1 <- summary(Model_1)
sum2 <- summary(Model_2)
sum3 <- summary(Model_3)

# Print full summaries (includes coefficient significance)
cat("\n Model 1\n")
print(sum1)

cat("\n Model 2 \n")
print(sum2)

cat("\n Model 3 \n")
print(sum3)

# Extract R² and Adjusted R² for each model
metrics <- data.frame(
  Model      = c("Model_1", "Model_2", "Model_3"),
  R_squared  = c(sum1$r.squared,
                 sum2$r.squared,
                 sum3$r.squared),
  Adj_R2     = c(sum1$adj.r.squared,
                 sum2$adj.r.squared,
                 sum3$adj.r.squared)
)

cat("\n R-squared Comparison \n")
print(metrics)

# Identify “best” model by highest Adjusted R²
best_idx   <- which.max(metrics$Adj_R2)
best_name  <- metrics$Model[best_idx]
cat("\nBest model by Adjusted R-squared is:", best_name, "\n")

# The coefficients & p-values for the best model only:
best_summary <- switch(
  best_name,
  "Model_1" = sum1,
  "Model_2" = sum2,
  "Model_3" = sum3
)
cat("\n Coefficients & Significance for", best_name, "\n")
print(best_summary$coefficients)


# Regression Tree – College GPA 
suppressWarnings(RNGversion("3.5.3"))

# Install packages once if needed
if (!require(caret))       install.packages("caret")
if (!require(rpart))       install.packages("rpart")
if (!require(rpart.plot))  install.packages("rpart.plot")
if (!require(forecast))    install.packages("forecast")

library(caret)
library(rpart)
library(rpart.plot)
library(forecast)

set.seed(1)

# 1. Read cleaned university data
myData <- read.csv("clean_university_data.csv")

# Keep only rows with observed College_GPA
myData_reg <- subset(myData, !is.na(College_GPA))

# 2. Train / validation split (70/30) on College_GPA
myIndex   <- createDataPartition(myData_reg$College_GPA, p = 0.70, list = FALSE)
trainSet  <- myData_reg[myIndex, ]
validSet  <- myData_reg[-myIndex, ]   

# 3. Default regression tree
set.seed(1)
default_tree <- rpart(
  College_GPA ~ HSGPA + SAT_ACT + Admitted + Enrolled + Edu_Parent1 + Edu_Parent2,
  data   = trainSet,
  method = "anova"
)
summary(default_tree)
prp(default_tree, type = 1, extra = 1, under = TRUE, main = "Default Regression Tree – College GPA")

# 4. Full tree (no pruning yet)
set.seed(1)
full_tree <- rpart(
  College_GPA ~ HSGPA + SAT_ACT + Admitted + Enrolled + Edu_Parent1 + Edu_Parent2,
  data      = trainSet,
  method    = "anova",
  cp        = 0,
  minsplit  = 2,
  minbucket = 1
)

prp(full_tree, type = 1, extra = 1, under = TRUE, main = "Full Regression Tree – College GPA")

# 5. Choose best cp from complexity-parameter (cp) table
printcp(full_tree)
best_cp <- full_tree$cptable[which.min(full_tree$cptable[, "xerror"]), "CP"]
cat("Best cp for regression tree:", best_cp, "\n")

# 6. Pruned tree using best cp
pruned_tree <- prune(full_tree, cp = best_cp)
prp(pruned_tree, type = 1, extra = 1, under = TRUE, main = "Pruned Regression Tree – College GPA")

# 7. Accuracy on validation set (continuous target)
predicted_gpa <- predict(pruned_tree, newdata = validSet)

# forecast::accuracy gives RMSE, MAE, MAPE, etc.
tree_metrics <- accuracy(predicted_gpa, validSet$College_GPA)
tree_metrics



