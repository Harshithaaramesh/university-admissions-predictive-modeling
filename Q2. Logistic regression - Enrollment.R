# Q2 – Enrollment Modeling
if (!require(caret))       
if (!require(rpart))       
if (!require(rpart.plot))  
if (!require(gains))       

library(caret)
library(rpart)
library(rpart.plot)
library(gains)

set.seed(1)

# 1) Load dataset with Admit_Tree_Prob from Q1
data2 <- read.csv("clean_university_with_AdmitProb.csv", stringsAsFactors = FALSE)

# Clean College text and standardize "Math & Science"
data2$College <- trimws(data2$College)
data2$College[grepl("Math *& *Science", data2$College)] <- "Math & Science"

# 2) Convert variables to proper types
data2$Enrolled <- factor(data2$Enrolled, levels = c(0, 1))
data2$Admitted <- factor(data2$Admitted, levels = c(0, 1))

# Categorical predictors
data2$College <- factor(data2$College)
data2$Sex     <- factor(data2$Sex)
data2$White   <- factor(data2$White)
data2$Asian   <- factor(data2$Asian)

# 3) Train / Validation Split (Enrollment logistic model)
n       <- nrow(data2)
n_train <- floor(0.7 * n)

TData2 <- data2[1:n_train, ]
VData2 <- data2[(n_train + 1):n, ]

# Align factor levels in validation with training
VData2$College <- factor(VData2$College, levels = levels(TData2$College))
VData2$Sex     <- factor(VData2$Sex,     levels = levels(TData2$Sex))
VData2$White   <- factor(VData2$White,   levels = levels(TData2$White))
VData2$Asian   <- factor(VData2$Asian,   levels = levels(TData2$Asian))

# 4) Logistic Regression for Enrollment (ModelEnr)
ModelEnr <- glm(
  Enrolled ~ HSGPA + SAT_ACT + Sex + White + Asian + Edu_Parent1 + Edu_Parent2 + College + Admit_Tree_Prob,
  family = binomial,
  data   = TData2
)

cat("\n Logistic Regression – Enrollment \n")
print(summary(ModelEnr))

# 5) Predictions on validation set 
pHatEnr <- predict(ModelEnr, VData2, type = "response")
yHatEnr <- ifelse(pHatEnr >= 0.5, 1, 0)

acc_enr <- mean(VData2$Enrolled == yHatEnr)
cat(sprintf("\nEnrollment logistic model accuracy = %.2f%%\n", 100 * acc_enr))

# Append probabilities and predicted class to validation rows
VData2$Enroll_Prob <- pHatEnr
VData2$Enroll_Pred <- yHatEnr

write.csv(VData2, "predicted_enrollment_dataset.csv", row.names = FALSE)
cat("Saved: predicted_enrollment_dataset.csv\n")

# 6. Classification Tree for Enrollment (Model 2 tree)
#    (Using same predictors + Admit_Tree_Prob)
# Use admitted-only records if that’s your Q2 definition:
dataQ2 <- subset(data2, Admitted == 1)

dataQ2$Enrolled_Factor <- factor(dataQ2$Enrolled, levels = c(0,1))

set.seed(3)
myIndex2 <- createDataPartition(dataQ2$Enrolled_Factor, p = 0.7, list = FALSE)
trainSet2 <- dataQ2[myIndex2, ]
validationSet2 <- dataQ2[-myIndex2, ]

# Fix factor levels in validation
validationSet2$College <- factor(validationSet2$College, levels = levels(trainSet2$College))
validationSet2$Sex     <- factor(validationSet2$Sex,levels = levels(trainSet2$Sex))
validationSet2$White   <- factor(validationSet2$White,levels = levels(trainSet2$White))
validationSet2$Asian   <- factor(validationSet2$Asian,levels = levels(trainSet2$Asian))

# Default tree
default_tree_enr <- rpart(
  Enrolled_Factor ~ HSGPA + SAT_ACT + Sex + White + Asian +
    Edu_Parent1 + Edu_Parent2 + College + Admit_Tree_Prob,
  data   = trainSet2,
  method = "class"
)
prp(default_tree_enr, type = 1, extra = 1, under = TRUE,main = "Default Tree – Enrollment")

# Full tree
full_tree_enr <- rpart(Enrolled_Factor ~ HSGPA + SAT_ACT + Sex + White + Asian +Edu_Parent1 + Edu_Parent2 + College + Admit_Tree_Prob,
  data   = trainSet2,
  method = "class",
  cp = 0,
  minsplit = 2,
  minbucket = 1
)
prp(full_tree_enr, type = 1, extra = 1, under = TRUE,
    main = "Full Tree – Enrollment")

# CP table & best cp
printcp(full_tree_enr)
best_cp_enr <- full_tree_enr$cptable[which.min(full_tree_enr$cptable[,"xerror"]), "CP"]
cat("Best cp for Enrollment tree:", best_cp_enr, "\n")

# Pruned tree
pruned_tree_enr <- prune(full_tree_enr, cp = best_cp_enr)
prp(pruned_tree_enr, type = 1, extra = 1, under = TRUE,
    main = "Pruned Tree – Enrollment")

# Performance on validation set
pred_class_enr <- predict(pruned_tree_enr, validationSet2, type = "class")
cm_enr <- confusionMatrix(pred_class_enr, validationSet2$Enrolled_Factor, positive = "1")
cm_enr

# Gains table
validationSet2_num <- validationSet2
validationSet2_num$Enrolled_num <- as.numeric(as.character(validationSet2_num$Enrolled_Factor))

pred_prob_enr_val <- predict(pruned_tree_enr, validationSet2_num, type = "prob")
gains(validationSet2_num$Enrolled_num, pred_prob_enr_val[,2])

# Append enrollment probabilities to admitted-only data
enr_prob_full  <- predict(pruned_tree_enr, dataQ2, type = "prob")[,2]
enr_class_full <- predict(pruned_tree_enr, dataQ2, type = "class")

dataQ2$Enroll_Tree_Pred <- enr_class_full
dataQ2$Enroll_Tree_Prob <- enr_prob_full

write.csv(dataQ2, "admitted_with_EnrollProb.csv", row.names = FALSE)
cat("Saved: admitted_with_EnrollProb.csv\n")

