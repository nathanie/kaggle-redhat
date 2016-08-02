#this script yields 0.95852 auc in the public LB

library(data.table)
library(xgboost)
library(caret)
library(dplyr)

setwd("c:/kaggle/redhat")
peopleData<- fread("people.csv",stringsAsFactors = F,data.table = F)
trData<- fread("act_train.csv",stringsAsFactors = F,data.table = F)
teData<- fread("act_test.csv",stringsAsFactors = F,data.table = F)
trData$source<- 1
teData$source<- 0

teData$outcome<- -1
all_data<- rbind(trData,teData)
all_data<- merge(x = all_data,y = peopleData,by = 1,all.x = T)

#-------------from this part i go on to initial submission XGBoost-------
#-------------in order to use different methods one hot encoding sould be used------
feature.names <- names(all_data)[c(3:14,17:56)]


for (f in feature.names) {
  if (class(all_data[[f]])=="character" | class(all_data[[f]])=="logical") {
    levels <- unique(all_data[[f]])
    all_data[[f]] <- as.integer(factor(all_data[[f]], levels=levels))
  }
}

full_train<- all_data[all_data$source==1,]
full_train$date.x<- as.numeric(full_train$date.x)
full_test<- all_data[all_data$source==0,]
full_test$date.x<- as.numeric(full_test$date.x)

temp<-nearZeroVar(full_train,99/1)
set.seed(12345)
inTrain<- createDataPartition(full_train$outcome,p = 0.8,list = F)

dtrain<-xgb.DMatrix(as.matrix(full_train[inTrain,feature.names]),label = full_train$outcome[inTrain])
dval<- xgb.DMatrix(as.matrix(full_train[-inTrain,feature.names]),label = full_train$outcome[-inTrain])

watchlist<-list(val=dval,train=dtrain)

param <- list(  objective           = "binary:logistic",
                "eval_metric"       = "auc",
                booster             = "gbtree",
                eta                 = 0.03, # 0.06, #0.01,
                max_depth           = 17, #changed from default of 8
                subsample           = 0.8, # 0.8
                colsample_bytree    = 0.8, # 0.7
                alpha               = 0, 
                min.child.weight    = 100
                # lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 300, #300, #280, #125, #250, # changed from 300
                    verbose             = 1,
                    early.stop.round    = 30,
                    watchlist           = watchlist,
                    maximize            = TRUE,
                    #feval               = RMSLE,
                    print.every.n       = 1
)

pred = predict(clf,as.matrix(full_test[,feature.names]))

full_test$pred<- pred

submission<- full_test %>%
  group_by(activity_id) %>%
  summarise(outcome = mean(pred))

write.csv(submission,"xgb_initial_submission.csv",row.names = F)
