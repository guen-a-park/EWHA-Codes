rm(list = ls())

install.packages("randomForest")
library(randomForest)
library(rpart) 

# data(stagec)	 
# str(stagec)

# stagec1<- subset(stagec, !is.na(g2))
# stagec2<- subset(stagec1, !is.na(gleason))
# stagec3<- subset(stagec2, !is.na(eet))
# str(stagec3)

getwd()
setwd("C:/Users/kate1/Desktop")

aba <- read.csv("UCI_heart_disease.csv", header=T)
names(aba)[1] <- c("age")

#target 값을 factor형으로 바꾸어줌
aba$target <- factor(aba$target)

set.seed(1234)

index <- sample(2, nrow(aba), replace=TRUE, prob=c(0.7, 0.3))
trainData <- aba[index==1,]
testData <- aba[index==2,]

# #Decision tree 생성
# heart_ctree <- ctree(target ~ age+sex+cp+trestbps+chol+fbs+restecg+thalach+exang+oldpeak+slope+ca+thal, data=trainData)
# 
# 
# print(heart_ctree)
# 
# #Decision tree plotting
# plot(heart_ctree)
# plot(heart_ctree, type="simple")
# 
# #Classificatioin with test set
# testPred1 <- predict(heart_ctree, newdata = testData)


rf <- randomForest(target ~ ., data=trainData, ntree=100, proximity=TRUE)	
# 반응변수(class)는  상동염색체수(ploidy)
#예측변수(속성)는  7개임
# proximity=TRUE는  개체들  간의  근접도  행렬을  제공:
# 동일한  최종노드에 포함되는  빈도에  기초함

#트리수에 따른 범주별 오분류율
#검은색은 전체 오분류율
plot(rf)
 
#테스트 자료에 대해 예측 수행

rf.pred <- predict(rf, newdata=testData)
tb <- table(rf.pred, testData$target)

#오분류율 계산
error.rpart <- 1-(sum(diag(tb))/sum(tb))
error.rpart 

#정확도를 백분율로 표현
sum(rf.pred==testData$target)/length(rf.pred)*100

#정확도 : 82.14286

