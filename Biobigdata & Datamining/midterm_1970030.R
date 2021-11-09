rm(list = ls())
#install.packages("party")

library(party)

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


#compute Shannon entropy
entropy <- function(target) {
  freq <- table(target)/length(target)
  # vectorize
  vec <- as.data.frame(freq)[,2]
  #drop 0 to avoid NaN resulting from log2
  vec<-vec[vec>0]
  #compute entropy
  -sum(vec * log2(vec))
}

#Decision tree 생성
heart_ctree <- ctree(target ~ age+sex+cp+trestbps+chol+fbs+restecg+thalach+exang+oldpeak+slope+ca+thal, data=trainData)


print(heart_ctree)

#Decision tree plotting
plot(heart_ctree)
plot(heart_ctree, type="simple")

#Classificatioin with test set
testPred1 <- predict(heart_ctree, newdata = testData)

#분류결과 plotting
testPred1
plot(testPred1)
table(testPred1, testData$target)

#예측 결과와 실제 데이터의 정확도 확인
sum(testPred1==testData$target)/length(testPred1)*100
