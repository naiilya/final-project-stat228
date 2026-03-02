## Introduction

Welcome to the “Data Wrangling” lesson, dear R users!

In this project, we work with the Lending Club Loan Dataset, extracted from Kaggle.

Lending Club is a peer-to-peer lending company based in the United States, connecting individuals who want to invest money with those looking to borrow. The Lending Club dataset contains loan data for loans issued across the U.S. between 2007 and 2015.

This large dataset, named “loans.csv”, offers real-world financial data on borrower profiles and their repayment outcomes. 
It contains approximately 2.26 million observations and 145 variables, covering loan details, borrower characteristics, account histories, and repayment statuses.

For this project, we focus on a specific subset of the dataset, narrowing in on the years of the Global Financial Crisis (2007, 2008, 2009). 
I extracted this subset directly on Kaggle, because working with the full dataset can overwhelm a standard CPU due to its enormous size. My smaller dataset, named “loans_gfc”, spans from 2007 to 2009 and contains 8,277 observations and 145 columns.

## Variables of Interest

Response Variable: 

- "loan_status" - borrower's loan status with four categories: 
“Fully Paid”, “Charged Off”, 
"Does not meet the credit policy. Status: Fully Paid", "Does not meet the credit policy. Status: Charged Off".

Explanatory Variables:

 - loan_amnt - amount the borrower applied for (requested principal) (USD);
 - funded_amnt - actual amount funded by investors (USD);
 - grade - lending Club–assigned credit grade (A–G);
 - int_rate - annual interest rate set on the loan (%);
 - installment - borrower’s fixed monthly repayment amount (USD);
 - emp_length - borrower’s reported employment length (e.g., “10+ years”, “<1 year”)
 - annual_inc - borrower’s self-reported annual income (USD);
 - dti - debt-to-income ratio, showing how leveraged the borrower is (%); 
 - open_acc - number of currently open credit lines;
 - revol_bal - total revolving balance (mainly credit cards) (USD);
 - revol_util - revolving credit utilization rate (%);
 - mths_since_last_delinq - months since the last delinquency (missing if none recorded)
 - verification_status - whether the borrower’s income was verified (e.g., “Verified”, “Not Verified”);
 
- Other Variables of Interest:
  - issue_d - loan issue date (Month-YYYY)
  - term - length of the loan term (e.g., 36 or 60 months);
  - home_ownership - borrower’s homeownership status (e.g., “RENT”, “OWN”);
  - purpose - reason for the loan (e.g., “debt consolidation”, “small business”);
  - total_acc - the total number of credit accounts the borrower has.

## Project Goal 

This project aims to predict whether borrowers will fully repay their loans or default using Lending Club data from the global financial crisis period. I apply random forest and logistic regression models to generate these predictions. 

The project goal can be framed as a set of two hypotheses:

- Null Hypothesis 1: The logistic regression model’s prediction accuracy, measured by the AUC (area under the curve) metric, is 0.5 or lower, indicating performance no better than random chance.

- Alternative Hypothesis 1: The logistic regression model’s prediction accuracy, measured by the AUC metric, exceeds 0.5, demonstrating meaningful predictive power.

- Null Hypothesis 2: The random forest model’s prediction accuracy, measured by the AUC metric, is 0.5 or lower, indicating performance no better than random chance.

- Alternative Hypothesis 2: The random forest model’s prediction accuracy, measured by the AUC metric, exceeds 0.5, demonstrating meaningful predictive power.


## Data Wrangling

We first need to load relevant packages for data analysis. Packages contain all the essential functions that make it easier to manipulate, clean, analyze, and visualize data.

```{r setup, include=TRUE, message=FALSE, warning=FALSE}
library(tidyverse)
library(tidytext)
library(text2vec)
library(randomForest)
library(caret)
library(pROC)
library(tm)
library(Matrix)
library(rpart)
library(rpart.plot)
library(learnr)
# Using "set.seed" command to make the results reproducible across different runs
set.seed(352)
```

```{r, include=TRUE, message=FALSE, warning=FALSE}
# Reading the CSV file
df_1 <- read.csv("loans_gfc.csv")
# View(df_1)

# Deleting empty columns - all the columns after the column "annual_inc_joint"
start_col <- which(names(df_1) == "annual_inc_joint")

df_2 <- df_1[, 1:(start_col - 1)]
```

## Explorotary Data Analysis

```{r, include=TRUE, message=FALSE, warning=FALSE}
# Selecting all the irrelevant columns:
cols_to_remove <- c(
  "addr_state", "application_type", "collection_recovery_fee", "collections_12_mths_ex_med", 
  "delinq_2yrs", "desc", "earliest_cr_line", "earliest_cr_line", "emp_title", 
  "funded_amnt_inv", "id", "initial_list_status", "inq_fi", "inq_last_12m", 
  "inq_last_6mths", "last_credit_pull_d", "last_credit_pull_d",
  "last_pymnt_amnt", "last_pymnt_d", "member_id", 
  "mths_since_last_major_derog", "mths_since_last_record", "next_pymnt_d", 
  "out_prncp", "out_prncp_inv", "policy_code", "pub_rec", "pymnt_plan", 
  "recoveries", "sub_grade", "title", "total_pymnt", "total_pymnt_inv",
  "total_rec_int", "total_rec_late_fee", "total_rec_prncp", "url", "zip_code"
)

df_3 <- df_2 |>
  # Deleting irrelevant columns
  select(-any_of(cols_to_remove)) |>
  
  # Converting specific variables to factors
  mutate(
    emp_length = as.factor(emp_length),
    home_ownership = as.factor(home_ownership),
    grade = as.factor(grade),
    
    # Changing the format of a binary variable: 1 = Verified, 0 = Not Verified
    verification_status = ifelse(verification_status == "Verified", 1, 0),
    
    # Changing the format of a quantitative variable: replacing NA values with 0
    mths_since_last_delinq = ifelse(is.na(mths_since_last_delinq), 0, mths_since_last_delinq)) |>
  
  # Shifting "issue_d" to first column, "loan_status" to second column
  select(issue_d, loan_status, everything()) |>
  
  # Omitting remaining missing values in this dataset
  na.omit()

df_3
# View(df_3)
```

```{r}
# Let's look at the loan amount distribution by loan status using "ggplot"!

ggplot(df_3, aes(x = loan_status, y = loan_amnt, fill = loan_status)) +
  geom_boxplot(outlier.alpha = 1, width = 0.6) +
  labs(
    title = "Loan Amount Distribution by Loan Status",
    x = "Loan Status",
    y = "Loan Amount (USD)"
  ) +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text.x = element_text(angle = 0, vjust = 0.5, face = "bold"),
    axis.text.y = element_text(face = "bold")
  )
```

This boxplot shows us that "Fully Paid" loans have a median loan amount of approximately 8,000, with most loans ranging between 5,000 and 13,000. Also, "Charged Off" loans show a higher median of about 10,000.

Loans categorized as “Does not meet the credit policy (Fully Paid)” have a much smaller median near 6,000, whereas “Does not meet the credit policy (Charged Off)” loans show a median around 7,000, slightly higher than their Fully Paid counterpart, but still lower than the main categories. 

Overall, this boxplot suggests that larger loans a slightly higher risk of default, though small loans can also result in charge-offs. 

In order to perform random forest and logistic regression, we need to covert the response variable 'loan_status' into a binary variable. 

I am not overwritting the original variable, I simply created a new response variable 'loan_status_binary'.

```{r}
data <- df_3 |>
  mutate(
    loan_status_binary = case_when(
      loan_status %in% c("Charged Off", "Does not meet the credit policy. Status:Charged Off") ~ "default",
      loan_status %in% c("Fully Paid", "Does not meet the credit policy. Status:Fully Paid") ~ "paid",
       TRUE ~ NA_character_
    )
  )

# Ensuring loan_status_binary is a factor with levels "default" and "paid"
data <- data |>
  mutate(loan_status_binary = factor(loan_status_binary, levels = c("default", "paid")))
# View(data)
```

```{r}
# Let's look at the modified plot on loan amount distribution by loan status using "ggplot"!

ggplot(data, aes(x = loan_status_binary, y = loan_amnt, fill = loan_status_binary)) +
  geom_boxplot(outlier.alpha = 0.5, width = 0.6) +
  labs(
    title = "Loan Amount Distribution by Loan Status",
    x = "Loan Status",
    y = "Loan Amount (USD)"
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text.x = element_text(angle = 0, vjust = 0.5, face = "bold"),
    axis.text.y = element_text(face = "bold")
  )
```

The new boxplot shows the distribution of loan amounts (in USD) across two binary loan outcomes: default and paid. Loans that defaulted have a median loan amount of approximately 9,000, with most values ranging between 5,000 and 15,000. 

Paid loans have a slightly lower median near 8,000, with most values spread between 5,000 and 12,000. However, paid loans display numerous high-value outliers reaching above $25,000. 

Overall, defaulted loans tend to have slightly larger typical amounts than paid loans, though the presence of high outliers in the "Paid" category highlights that even large loans can be successfully repaid.

## Dataframes Preparation

```{r}
# Creating 80/20 dataframes
train_indexes <- as.vector(createDataPartition(data$loan_status_binary, p = 0.8, list = FALSE))
data_train <- slice(data, train_indexes)
data_test <- slice(data, -train_indexes)
```

For the further analysis, I'm splitting the orginal dataframe into training and testing dataframes.

## Logistic Regression for Predicting Loan Repayment Outcomes

```{r}
# Fitting logistic regression model
fit <- glm(
  loan_status_binary ~ loan_amnt + funded_amnt + grade + 
    int_rate + installment + emp_length + annual_inc + 
    dti + mths_since_last_delinq + verification_status + 
    open_acc + revol_bal + revol_util,
  data = data_train,
  family = "binomial"
  )
```

```{r, include=TRUE, message=FALSE, warning=FALSE}
# Selecting an arbitrary threshold
threshold <- 0.1

# Predicting probabilities and classifying on training set
predicted_prob_train <- predict(fit, newdata = data_train, type = "response")
predicted_class_train <- as.factor(if_else(predicted_prob_train > threshold, "default", "paid"))

# Predicting probabilities and classifying on testing set
predicted_prob_test <- predict(fit, newdata = data_test, type = "response")
predicted_class_test <- as.factor(if_else(predicted_prob_test > threshold, "default", "paid"))
```

```{r, include=TRUE, message=FALSE, warning=FALSE}
# Creating confusion matrices
conf_matrix_train <- confusionMatrix(
  data = predicted_class_train,
  reference = data_train$loan_status_binary,
  positive = "default"
)
conf_matrix_train

conf_matrix_test <- confusionMatrix(
  data = predicted_class_test,
  reference = data_test$loan_status_binary,
  positive = "default"
)
conf_matrix_test

print(conf_matrix_train$byClass["Sensitivity"])
print(conf_matrix_test$byClass["Sensitivity"])
```

```{r}
# Creating ROC objects
roc_train <- roc(data_train$loan_status_binary, predicted_prob_train, levels = c("paid", "default"))
roc_test <- roc(data_test$loan_status_binary, predicted_prob_test, levels = c("paid", "default"))

# Visualizing ROC curves
plot(roc_train, col = "black", lwd = 2, main = "ROC curve on cleaned training and testing data")
plot(roc_test, col = "darkred", lwd = 2, add = TRUE)

# Adding a legend
legend("bottomright",
  legend = c(
    paste("Cleaned Train AUC =", round(auc(roc_train), 5)),
    paste("Cleaned Test AUC =", round(auc(roc_test), 5))
  ),
  col = c("black", "darkred"),
  lwd = 2
)

# Getting AUC values
print(paste("Training AUC:", auc(roc_train)))
print(paste("Testing AUC:", auc(roc_test)))
```


Output Values:

 - Training AUC: 0.662930329429786
 - Testing AUC: 0.643413277719847

The logistic regression model’s performance, measured by the area under the ROC curve (AUC), shows meaningful predictive power on both the training and testing datasets. 

On the training data, the model achieves an AUC of approximately 0.663, indicating that it correctly distinguishes between defaulted and paid loans about 66% of the time.

On the testing data, the AUC is slightly lower at around 0.643, suggesting the model generalizes reasonably well to unseen data.

Overall, these results confirm that the logistic regression model captures important patterns in the data, although there remains room for improvement, possibly through more complex models or additional meaningful explanatory variables.


## Decision Tree for Predicting Loan Repayment Outcomes

```{r}
# Let's start with a single decision tree:
tree_fit <- rpart(
    loan_status_binary ~ loan_amnt + funded_amnt + grade + 
    int_rate + installment + emp_length + annual_inc + 
    dti + mths_since_last_delinq + verification_status + 
    open_acc + revol_bal + revol_util,
  data = data_train,
  method = "class"
)
```

```{r, include=TRUE, message=FALSE, warning=FALSE}
# Predicting probabilities
predicted_prob_train_tree <- predict(tree_fit, newdata = data_train, type = "prob")[, "default"]
predicted_prob_test_tree <- predict(tree_fit, newdata = data_test, type = "prob")[, "default"]
```

```{r}
# Creating ROC objects
roc_train_tree <- roc(data_train$loan_status_binary, predicted_prob_train_tree, levels = c("paid", "default"))
roc_test_tree <- roc(data_test$loan_status_binary, predicted_prob_test_tree, levels = c("paid", "default"))

# Visualizing ROC curves
plot(roc_train_tree, col = "blue", lwd = 2, main = "ROC curve for single tree (cleaned data)")
plot(roc_test_tree, col = "purple", lwd = 2, add = TRUE)

# Adding a legend
legend("bottomright",
  legend = c(
    paste("Tree Train AUC =", round(auc(roc_train_tree), 5)),
    paste("Tree Test AUC =", round(auc(roc_test_tree), 5))
  ),
  col = c("blue", "purple"),
  lwd = 2
)
```

The single decision tree model’s performance shows no meaningful predictive ability on either the training or testing datasets. 

Specifically, the model yields a training AUC of 0.5 and a testing AUC of 0.5, meaning its predictions are no better than random chance. It fails to discriminate between defaulted and paid loans. 

These weak results were expected, because a single decision tree often lacks the complexity and robustness needed to handle large and noisy datasets. To improve predictive performance, we turn to the random forest approach, which aggregates many decision trees to reduce variance and avoid overfitting.


## Random Forest Predicting Loan Repayment Outcomes

```{r}
# Fitting a random forest
rf_fit <- randomForest(
    loan_status_binary ~ loan_amnt + funded_amnt + grade + 
    int_rate + installment + emp_length + annual_inc + 
    dti + mths_since_last_delinq + verification_status + 
    open_acc + revol_bal + revol_util,
  data = data_train,
  ntree = 500,
  importance = TRUE
)
```

```{r, include=TRUE, message=FALSE, warning=FALSE}
# Predicting probabilities
predicted_prob_train_rf <- predict(rf_fit, newdata = data_train, type = "prob")[, "default"]
predicted_prob_test_rf <- predict(rf_fit, newdata = data_test, type = "prob")[, "default"]

# Creating ROC objects
roc_train_rf <- roc(data_train$loan_status_binary, predicted_prob_train_rf, levels = c("paid", "default"))
roc_test_rf <- roc(data_test$loan_status_binary, predicted_prob_test_rf, levels = c("paid", "default"))

# Visualizing ROC curves
plot(roc_train_rf, col = "black", lwd = 2, main = "ROC curve for random forest (cleaned data)")
plot(roc_test_rf, col = "darkorange", lwd = 2, add = TRUE)

# Adding a legend
legend("bottomright",
  legend = c(
    paste("RF Train AUC =", round(auc(roc_train_rf), 5)),
    paste("RF Test AUC =", round(auc(roc_test_rf), 5))
  ),
  col = c("black", "darkorange"),
  lwd = 2
)
```

```{r}
# Getting AUC values
print(paste("Random Forest Training AUC:", auc(roc_train_rf)))
print(paste("Random Forest Testing AUC:", auc(roc_test_rf)))

print(importance(rf_fit))
```

The random forest model’s performance shows strong evidence of overfitting on the training data (100% accuracy) and more moderate generalization on the testing set. 

Specifically, the random forest achieves a perfect training AUC of 1, meaning it completely separates defaulted and paid loans on the training data. Although the model perfectly memorizes the training data, it fails to generalize as effectively to unseen cases. On the testing data, the model achieves an AUC of approximately 0.627, indicating that it correctly distinguishes between defaulted and paid loans 63% of the time. 

For this analysis, I used 500 decision trees trained on approximately 8,000 observations, allowing the model to average across many weak learners and reduce variance. The variable importance output reveals that among the predictors, revol_bal, revol_util, dti, annual_inc, and installment rank as the most influential features, with high MeanDecreaseGini and MeanDecreaseAccuracy values. Notably, loan-specific terms like int_rate, funded_amnt, and loan_amnt also show strong contributions, confirming their critical role in predicting loan outcomes. 

Overall, the random forest’s ability to combine multiple trees gives it a clear advantage over single decision tree. These results confirm that the random forest model captures important patterns in the data, although there remains room for improvement, possibly through more complex models or additional meaningful explanatory variables.

## Conclusion 

In conclusion, this project applied both logistic regression and random forest models to predict loan repayment outcomes using Lending Club data from the global financial crisis period. Despite the random forest’s complex ensemble approach, logistic regression performed slightly better, achieving a testing AUC of approximately 0.64, compared to the random forest’s testing AUC of approximately 0.63.

Hypotheses Results: we can reject the first null hypothesis, because our logistic regression model with AUC = 0.64 is above random chance (0.5); and we can also reject the second null hypothesis, because our random forest model with AUC = 0.63 is above random chance (0.5).

## References

Adarsh S. (2021). Lending Club Loan Data (CSV). Kaggle : https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv
