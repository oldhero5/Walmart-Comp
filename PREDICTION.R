##XGB DATA PREDICT
# Set WD
setwd("~/DATA/KAGGLE/Walmart")

#Librarys 
library(reshape2)
library(data.table)
library(xgboost)
library(Rtsne)
library(caret)
library(ggplot2)
library(readr)
library(lubridate)

#Read Data
train <- read_csv("train.csv")
test  <- read_csv("test.csv")
samsub <- read_csv("sample_submission.csv")

#Clean Data
train1 <- train
test1 <- test
train1[is.na(train1)]   <- 0
test1[is.na(test1)]   <- 0

train$Weekday<- make.names(train$Weekday)
train$Weekday <- as.factor(train$Weekday)
train$DepartmentDescription<- make.names(train$DepartmentDescription)
train$DepartmentDescription <- as.factor(train$DepartmentDescription)

train1$TripType <-paste0("TripType_",train1$TripType)
train1$TripType <- as.factor(train1$TripType)

train1$Monday<- as.numeric(train$Weekday == "Monday")
train1$Tuesday<- as.numeric(train$Weekday == "Tuesday")
train1$Wednesday<- as.numeric(train$Weekday == "Wednesday")
train1$Thursday<- as.numeric(train$Weekday == "Thursday")
train1$Friday<- as.numeric(train$Weekday == "Friday")
train1$Saturday<- as.numeric(train$Weekday == "Saturday")
train1$Sunday<- as.numeric(train$Weekday == "Sunday")

test1$Monday<- as.numeric(test$Weekday == "Monday")
test1$Tuesday<- as.numeric(test$Weekday == "Tuesday")
test1$Wednesday<- as.numeric(test$Weekday == "Wednesday")
test1$Thursday<- as.numeric(test$Weekday == "Thursday")
test1$Friday<- as.numeric(test$Weekday == "Friday")
test1$Saturday<- as.numeric(test$Weekday == "Saturday")
test1$Sunday<- as.numeric(test$Weekday == "Sunday")

test$Weekday<- make.names(test$Weekday)
test$Weekday <- as.factor(test$Weekday)
test$DepartmentDescription<- make.names(test$DepartmentDescription)
test$DepartmentDescription <- as.factor(test$DepartmentDescription)

train1 <- dcast(train1, VisitNumber + TripType + Monday + Tuesday + Wednesday + Thursday + Friday +Saturday +Sunday ~ DepartmentDescription,fun.aggregate = sum, value.var = "ScanCount")
test1 <- dcast(test1, VisitNumber  + Monday + Tuesday + Wednesday + Thursday + Friday +Saturday +Sunday ~ DepartmentDescription,fun.aggregate = sum, value.var = "ScanCount")

# creates total items purchased
train1$TotalItems <- rowSums(train1[,c(5:73)])
test1$TotalItems <- rowSums(test1[,c(4:71)])

Finelinetrain <- dcast(train, VisitNumber + TripType ~ DepartmentDescription, fun.aggregate = function(x) length(unique(x)), value.var = "FinelineNumber")
Finelinetest <- dcast(test, VisitNumber ~ DepartmentDescription, fun.aggregate = function(x) length(unique(x)), value.var = "FinelineNumber")

Skutrain <- dcast(train, VisitNumber + TripType ~ DepartmentDescription, fun.aggregate = function(x) length(unique(x)), value.var = "Upc")
Skutest <- dcast(test, VisitNumber ~ DepartmentDescription, fun.aggregate = function(x) length(unique(x)), value.var = "Upc")

Deptstrain <- dcast(train, VisitNumber + TripType ~ DepartmentDescription, fun.aggregate = function(x) length(unique(x)), value.var = "DepartmentDescription")
Deptstest <- dcast(test, VisitNumber ~ DepartmentDescription, fun.aggregate = function(x) length(unique(x)), value.var = "DepartmentDescription")

train1$TotalFLines <- rowSums(Finelinetrain[,c(3:71)])
test1$TotalFLines <- rowSums(Finelinetest[,c(2:69)])

train1$TotalSku<- rowSums(Skutrain[,c(3:71)])
test1$TotalSku <- rowSums(Skutest[,c(2:69)])

train1$TotalDepts <- rowSums(Deptstrain[,c(3:71)])
test1$TotalDepts <- rowSums(Deptstest[,c(2:69)])



#remove targets
target.org <- train1$TripType
target <- target.org
levels(target)
num.class <- length(levels(target))
levels(target) <- 1:num.class
train1$TripType <- NULL
train1$VisitNumber <- NULL

#convert to matrix
train.mat <- as.matrix(train1)
colnames(train.mat) <- NULL
test.mat <- as.matrix(test1)
mode(train.mat) <- "numeric"
colnames(test.mat)<- NULL
mode(test.mat) <- "numeric"
train.mat <- train.mat[,-1]
train.mat <- train.mat[,-1]
test.mat <- test.mat[,-1]
y <- as.matrix(as.integer(target)-1)

#k-fold cross validation with time
param <- list("objective" = "binary:logitraw", "eval_matric" = "merror","num_class" = num.class, "nthread" = 38, "max_depth" = 38, "eta" = 0.3,"gamma" = 0, 
              "subsample" = 10, "colsample_bytree" = 10, "min_child_weight" = 12)
set.seed(1234)
#kfold cross validation w/ timing
nround.cv = 200

system.time(bst.cv <- xgb.cv(params = param, data = train.mat, label = y, nfold = 4, nrounds = nround.cv, prediction = TRUE, verbose = FALSE))

tail(bst.cv$dt)

#index of minimum merror
min.merror.idx <- which.min(bst.cv$dt[,test.merror.mean])
min.merror.idx

#minimum error
bst.cv$dt[min.merror.idx,]

#get CV's prediction decoding
pred.cv = matrix(bst.cv$pred, nrow=length(bst.cv$pred)/num.class, ncol=num.class)
pred.cv = max.col(pred.cv, "last")


#real model fit training, with full data
system.time(bst<- xgboost(param = param, data = train.mat, label = y, nrounds = min.merror.idx, verbose = 0))

#xgboost predict using test matrix
pred <- predict(bst, test.mat)

#deprocess
pred<- matrix(pred, nrow = num.class, ncol = length(pred)/num.class)
pred<- t(pred)
pred<- max.col(pred, "last")

#get trained model
model <- xgb.dump(bst, with.stats = TRUE)
#get real names
names <- dimnames(train1)[[2]]
#feature importance
importance_matrix <- xgb.importance(names, model = bst)
#plot feature importance
gp <- xgb.plot.importance(importance_matrix)
print(gp)

#Write file
f <- colnames(samsub)
f <- f[2:39]
f <- gsub("\\TripType_", "", f)
f <- as.integer(f)
G <- outer(pred,f, function(x,y) x==y)
G <- G*1
G <- as.data.table(G)
a <- samsub$VisitNumber
G <- cbind(a,G)
d <- colnames(samsub)
colnames(G) <- d
write_csv(G,"solution1.csv")

