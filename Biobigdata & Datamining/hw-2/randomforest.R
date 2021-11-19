rm(list = ls())

#���̺귯�� ��ġ�ϱ�
install.packages("randomForest")
library(randomForest)
library(rpart) 

getwd()
setwd("C:/Users/kate1/Desktop")

aba <- read.csv("UCI_heart_disease.csv", header=T)
names(aba)[1] <- c("age")

#target ���� factor������ �ٲٱ�
aba$target <- factor(aba$target)

set.seed(1234)

#train,test split
index <- sample(2, nrow(aba), replace=TRUE, prob=c(0.7, 0.3))
trainData <- aba[index==1,]
testData <- aba[index==2,]

#RF �� ����
rf <- randomForest(target ~ ., data=trainData, ntree=100, proximity=TRUE)	

# proximity=TRUE��  ��ü��  ����  ������  �����  ����:
# ������  ������忡 ���ԵǴ�  �󵵿�  ������
#Ʈ������ ���� ���ֺ� ���з���
#�������� ��ü ���з���
plot(rf)
 
#�׽�Ʈ �ڷῡ ���� ���� ����

rf.pred <- predict(rf, newdata=testData)
tb <- table(rf.pred, testData$target)

#���з��� ���
error.rpart <- 1-(sum(diag(tb))/sum(tb))
error.rpart 

#��Ȯ���� ������� ǥ��
sum(rf.pred==testData$target)/length(rf.pred)*100

#��Ȯ�� : 82.14286
