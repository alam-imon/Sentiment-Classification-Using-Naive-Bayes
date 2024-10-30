#Read the file in R
bank = read.csv("bank.csv", sep = ";", header = TRUE)


#Excluding 'duration' variable from the dataset
bank= bank[,-12]
summary(bank)


#If there any missing value in the dataset
sum(is.na(bank))
#Structure of the uploaded dataset
#library(plyr)
glimpse(bank)


#Coverting pdays variable into bins
bank$pdays = cut(bank$pdays, breaks = c(-2,-1,91,182,273,364,455,546,637,728,819,910),
                 labels = c('Not contacted','3months passed','6months passed',
                            '9months passed','1yr passed','1.3yrs passed',
                            '1.6yrs passed','1.9yrs passed','2yrs passed',
                            '2.3yrs passed','More than 2.3yrs passed'))


#Training & Test Datasets
deposit_yes <- bank[which(bank$y == 'yes'), ]  # all yes's of outcome class
deposit_no <- bank[which(bank$y == 'no'), ]  # all no's of outcome class
set.seed(100)
deposit_yes_training_rows <- sample(1:nrow(deposit_yes), 0.7*nrow(deposit_yes))  #randomly choosing 70% observations of yes class
deposit_no_training_rows <- sample(1:nrow(deposit_no), 0.7*nrow(deposit_no))  #randomly choosing 70% observations of yes class
training_yes <- deposit_yes[deposit_yes_training_rows, ]  
training_no <- deposit_no[deposit_no_training_rows, ]
trainingData <- rbind(training_yes, training_no)  #combining chosen observations
glimpse(trainingData)
table(trainingData$y)


# Create Test Data
test_yes <- deposit_yes[-deposit_yes_training_rows, ]
test_no <- deposit_no[-deposit_no_training_rows, ]
testData <- rbind(test_yes, test_no)  #combining chosen observations
glimpse(testData)
table(testData$y)

#Generalized Linear Model 

fitControl <- trainControl(## 10-fold CV
  method = "cv",
  number = 10,
  savePredictions = TRUE
)
#Fitting the model
glm.model = train(y~., data = trainingData, method="glm", family=binomial(),
                  trControl= fitControl)
summary(glm.model)
# Variable importance by the GLM model
varImp(glm.model)
#Predicting outcome variable on test set
y.pred= predict(glm.model, testData)
#Confusion Matrix
confusionMatrix(y.pred, testData$y)


#Gradient Boosting Machine

#Tuning the hyper paramters
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*30, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)
set.seed(825)
#10-fold cross validation
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)
#Fitting the model
gbmFit2 <- train(y ~ ., data = trainingData, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE,
                 tuneGrid = gbmGrid)

plot(gbmFit2)
#Variable importance by the model
varImp(gbmFit2)
#Predicting outcome variable on test data
y.gbm.pred = predict(gbmFit2, testData)
#Confusion matrix
confusionMatrix(y.gbm.pred, testData$y)


#K-Nearest Neighbors

#Normalizing continuous variables
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
bank$age = normalize(bank$age)
bank$balance = normalize(bank$balance)
bank$day = normalize(bank$day)
bank$campaign = normalize(bank$campaign)
bank$previous = normalize(bank$previous)

#Encoding categorical variabled to dummy. KNN uses distance metrics to identify classes 
library(dummies)
dummies <- dummyVars(~ ., data=bank[, -16])
c2 <- predict(dummies, bank[, -16])
bank.upd <- as.data.frame(cbind(bank$y, c2))
glimpse(bank.upd)
bank.upd$V1= as.factor(bank.upd$V1)
names(bank.upd)[which(names(bank.upd) == "V1")] <- "y"
bank.upd$y= factor(bank.upd$y, levels= c(1,2), labels= c("no","yes"))

#Training & Test Data after encoding variables
deposit_yes <- bank.upd[which(bank.upd$y == 'yes'), ]
deposit_no <- bank.upd[which(bank.upd$y == 'no'), ]
set.seed(100)
deposit_yes_training_rows <- sample(1:nrow(deposit_yes), 0.7*nrow(deposit_yes))
deposit_no_training_rows <- sample(1:nrow(deposit_no), 0.7*nrow(deposit_no))
training_yes <- deposit_yes[deposit_yes_training_rows, ]  
training_no <- deposit_no[deposit_no_training_rows, ]
trainingData <- rbind(training_yes, training_no) 
glimpse(trainingData)
table(trainingData$y)

# Create Test Data
test_yes <- deposit_yes[-deposit_yes_training_rows, ]
test_no <- deposit_no[-deposit_no_training_rows, ]
testData <- rbind(test_yes, test_no) 
glimpse(testData)
table(testData$y)

#Fitting the model
set.seed(156)
knn.mod = train(
  y ~ .,
  data = trainingData,
  method = "knn",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(k = seq(1, 30, by = 2))
)
plot(knn.mod)
#Best k-value chosen by KNN
knn.mod$bestTune
str(knn.mod)
# Variable importance by KNN
plot(varImp(knn.mod, top=15))
#Predicting outcome variable on test set
y.knn.pred = predict(knn.mod, testData)
#Confusion Matrix
confusionMatrix(y.knn.pred, testData$y)


#Support Vector Machine

trctrl <- trainControl(method = "cv", number = 10)
#Tuning hyper parameters
grid_radial <- expand.grid(sigma = c(0,0.01, 0.02, 0.025, 0.03, 0.04,
                                     0.05, 0.06, 0.07,0.08, 0.09, 0.1, 0.25, 0.5, 0.75,0.9),
                           C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75,
                                 1, 1.5, 2,5))
#Fitting the model
#Note, here training and test datasets have dummy encoded and normalized variables
set.seed(3233)
svm_Radial_Grid <- train(y ~., data = trainingData, method = "svmRadial",
                         trControl=trctrl,
                         tuneGrid = grid_radial,
                         tuneLength = 10)
svm_Radial_Grid
#Variable importance by SVM model
varImp(svm_Radial_Grid)
#Prediction the outcome variable on test data
y.radial.pred = predict(svm_Radial_Grid, testData)
#Confusion Matrix
confusionMatrix(y.radial.pred, testData$y)





#Naive Bayes

#Since in naive bayes algorithm, all variables have considered as categorical
#I am uploading the dataset again to avoid any confusion
#Read the file in R
bank = read.csv("bank.csv", sep = ";", header = TRUE)

#Excluding 'duration' variable from the dataset
bank= bank[,-12]

#Converting all continuous variables to categorical
bank$pdays = cut(bank$pdays, breaks = c(-2,-1,91,182,273,364,455,546,637,728,819,910),
                 labels = c('Not contacted','3months passed','6months passed',
                            '9months passed','1yr passed','1.3yrs passed',
                            '1.6yrs passed','1.9yrs passed','2yrs passed',
                            '2.3yrs passed','More than 2.3yrs passed'))
bank$balance= cut(bank$balance, breaks = c(-3314,40,332,851,1882,71188),
                  labels = c('balNegToNil','balNilToVLow','balVLowToLow',
                             'balLowToMed','balMedToHigh'))
bank$age= cut(bank$age, breaks = c(18,32,37,44,52,87),
              llabels = c('age19To32','age33To37','age38To44','age45To52',
                          'age53To87'))
bank$day= cut(bank$day, breaks = c(0,7,14,21,28,32), labels = c('Wk1','Wk2',
                                                                'Wk3','Wk4',
                                                                'Wk5'))
bank$campaign= cut(bank$campaign, breaks = c(0,1,2,60),
                   labels = c('curn1Time','curn2Times','curnMoreThan2Times'))
bank$previous= cut(bank$previous, breaks = c(-1, 0,1,3,30),
                   labels = c('preNotContacted','pre1Time','pre2-3Times',
                              'preMoreThan3Times'))
glimpse(bank)
#Training & Test Datasets
deposit_yes <- bank[which(bank$y == 'yes'), ]  # all yes's of outcome class
deposit_no <- bank[which(bank$y == 'no'), ]  # all no's of outcome class
set.seed(100)
deposit_yes_training_rows <- sample(1:nrow(deposit_yes), 0.7*nrow(deposit_yes))  #randomly choosing 70% observations of yes class
deposit_no_training_rows <- sample(1:nrow(deposit_no), 0.7*nrow(deposit_no))  #randomly choosing 70% observations of yes class
training_yes <- deposit_yes[deposit_yes_training_rows, ]  
training_no <- deposit_no[deposit_no_training_rows, ]
trainingData <- rbind(training_yes, training_no)  #combining chosen observations
glimpse(trainingData)
table(trainingData$y)

# Create Test Data
test_yes <- deposit_yes[-deposit_yes_training_rows, ]
test_no <- deposit_no[-deposit_no_training_rows, ]
testData <- rbind(test_yes, test_no)  #combining chosen observations

#Fitting the model
set.seed(128)
ctrl <- trainControl(method="cv", 10)
grid <- expand.grid(fL=c(0,0.5,1.0), usekernel = c(TRUE,FALSE), adjust=c(0,0.5,1.0))
x = trainingData[,-16] #matrix without outcome variable
y= trainingData$y
naive.model <- train(x, y, method="nb",
                     trControl=ctrl, tuneGrid = grid)
#Predicting outcome variable on test set
y.naive.pred = predict(naive.model, testData)
#Confusion matrix
confusionMatrix(y.naive.pred, testData$y)
#Variable importance by the naive model
plot(varImp(naive.model))
