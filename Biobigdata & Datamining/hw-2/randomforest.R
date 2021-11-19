rm(list = ls())

#라이브러리 설치하기
install.packages("randomForest")
library(randomForest)
library(rpart) 

getwd()
setwd("C:/Users/kate1/Desktop")

aba <- read.csv("UCI_heart_disease.csv", header=T)
names(aba)[1] <- c("age")

#target 값을 factor형으로 바꾸기
aba$target <- factor(aba$target)

set.seed(1234)

#train,test split
index <- sample(2, nrow(aba), replace=TRUE, prob=c(0.7, 0.3))
trainData <- aba[index==1,]
testData <- aba[index==2,]

#RF 모델 생성
rf <- randomForest(target ~ ., data=trainData, ntree=100, proximity=TRUE)	

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

