---
title: "ML - Project"
author: "Mahmoud_Osman"
date: "3/12/2020"
output:
  html_document: default
  pdf_document: default
---

```{r load myData, include=FALSE, echo=FALSE}
load("my_data.RData")
```

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Load Libraries

```{r LIBRARIES, warning=FALSE}
library(caret)
library(ggplot2)
library(randomForest)
```

## setting seed

```{r}
set.seed(159)
```

## Reading data:

```{r READING DATA}
training_dat<-read.csv('pml-training.csv',na.strings = c("NA", ""))
testing_dat<-read.csv('pml-testing.csv',na.strings = c("NA", ""))
```


# Subsetting data 70/30
```{r SUBSetting}
inTrain <- createDataPartition(y=training_dat$classe, p=0.70,list = F )
training <- training_dat[inTrain, ]
testing <- training_dat[-inTrain, ]
```

# Cleaning data:
## Remove Near Zero Variance data

```{r CLEANING_1}
Col_rm<-nearZeroVar(training)

training<-training[,-Col_rm]
testing<-testing[,-Col_rm]
```
## Remove columns with NA's more that 70% of data

```{r CLEANING_2}
Col_rm2<-which((sapply(training, function(x)sum(is.na(x))))>dim(training)[1]*0.7)
training<-training[,-Col_rm2]
testing<-testing[,-Col_rm2]
```
## Remove Personally Identifiable Information & Timestamp
```{r CLEANING_3}
training<-training[,-(1:5)]
testing<-testing[,-(1:5)]
```
## Applying the same for validation dataset:
```{r CLEANING_V}
test_valid <- testing_dat[,-Col_rm]
test_valid <- test_valid[,-Col_rm2]
test_valid <- test_valid[,-(1:5)]
```

# Prediction models
Multiple models (RF, LDA, RPART) are used for the prediciotn process:
```{r MODELS, CachedChunk, cache=TRUE, echo= FALSE, eval = FALSE}
mod_rf <- train(classe ~ ., data = training, method = "rf", allowParallel=T, ntree=50)
mod_lda <- train(classe ~ ., data = training, method = "lda")
mod_rpart <- train(classe ~ .,data=training,method="rpart")

pred_rf <- predict(mod_rf, testing)
pred_lda <- predict(mod_lda, testing)
pred_rpart <- predict(mod_rpart, testing)
```
## Combined model
```{r MOD-COMB, echo=FALSE, eval = FALSE}
predDF <- data.frame(pred_rf, pred_lda, pred_rpart, classe = testing$classe)
mod_comb <- train(classe ~ ., method = "rf", data = predDF,allowParallel=T, ntree=40)
pred_comb <- predict(mod_comb, predDF)
```
## Confusion matrix for each model
```{r CM}
CM_rf <- confusionMatrix(pred_rf, testing$classe)         #RF
print(CM_rf)
CM_lda <- confusionMatrix(pred_lda, testing$classe)       #lda
print(CM_lda)
CM_rpart <- confusionMatrix(pred_rpart, testing$classe)   #rpart
print(CM_rpart)
CM_comb <- confusionMatrix(pred_comb, testing$classe)      #combined
print(CM_comb)
```

## Plots of predictions vs observations

```{r PLOTS, echo=FALSE}
qplot(classe, pred_rf, data=testing,  colour= classe, geom = c("boxplot", "jitter"), 
      main = "RF predicted vs. observed in testing set", xlab = "Observed Classes", ylab = "Predicted Classes")

qplot(classe, pred_lda, data=testing,  colour= classe, geom = c("boxplot", "jitter"), 
      main = "LDA predicted vs. observed in testing set", xlab = "Observed Classes", ylab = "Predicted Classes")

qplot(classe, pred_rpart, data=testing,  colour= classe, geom = c("boxplot", "jitter"), 
      main = "RPART predicted vs. observed in testing set", xlab = "Observed Classes", ylab = "Predicted Classes")
```

Results show that RF is the best method despite using few trees.
Combined model have same accuracy as RF but it doesn't have any added value but more time complex to interpret so RF is chosen

# Validation Test
```{r VALIDATION}
Vpred_rf <- predict(mod_rf, test_valid)
print(Vpred_rf)
Vpred_lda <- predict(mod_lda, test_valid)
print(Vpred_lda)
Vpred_rpart <- predict(mod_rpart, test_valid)
print(Vpred_rpart)
```
