rm(list = ls())

library(randomForest)
library(rpart) 

data(stagec)	 
str(stagec)

stagec1<- subset(stagec, !is.na(g2))
stagec2<- subset(stagec1, !is.na(gleason))
stagec3<- subset(stagec2, !is.na(eet))
str(stagec3)

 
set.seed(1234)
ind <- sample(2, nrow(stagec3), replace=TRUE, prob=c(0.7, 0.3))

ind
trainData <- stagec3[ind==1, ]
testData <- stagec3[ind==2, ]

 


rf <- randomForest(ploidy ~ ., data=trainData, ntree=100, proximity=TRUE)	
# 반응변수(class)는  상동염색체수(ploidy)
#예측변수(속성)는  7개임
# proximity=TRUE는  개체들  간의  근접도  행렬을  제공:
# 동일한  최종노드에 포함되는  빈도에  기초함

#트리수에 따른 범주별 오분류율
#검은색은 전체 오분류율
plot(rf)
 
#테스트 자료에 대해 예측 수행

rf.pred <- predict(rf, newdata=testData)
tb <- table(rf.pred, testData$ploidy)

#오분류율 계산
error.rpart <- 1-(sum(diag(tb))/sum(tb))
error.rpart 

#정확도를 백분율로 표현
sum(rf.pred==testData$ploidy)/length(rf.pred)*100



