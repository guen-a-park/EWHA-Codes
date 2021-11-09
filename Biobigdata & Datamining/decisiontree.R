rm(list = ls())
#install.packages("party")

library(party)


# 아이리스 데이터
data(iris)
str(iris)  #structure
iris

set.seed(1234)

ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.6, 0.4))
trainData <- iris[ind==1,]
testData <- iris[ind==2,]

#Decision tree 생성
iris_ctree <- ctree(Species ~ Sepal.Length + Sepal.Width 
                    + Petal.Length + Petal.Width, data=trainData)
 

print(iris_ctree)

#Decision tree plotting
plot(iris_ctree)
plot(iris_ctree, type="simple")

#Classificatioin with test set
testPred <- predict(iris_ctree, newdata = testData)

#분류결과 plotting
testPred
plot(testPred)
table(testPred, testData$Species)

#예측 결과와 실제 데이터의 정확도 확인
sum(testPred==testData$Species)/length(testPred)*100



###### 전립선 암 데이터 ########

rm(list = ls())

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

tree <- ctree(ploidy ~ ., data=trainData)
tree
plot(tree)

testPred = predict(tree, newdata=testData)
table(testPred, testData$ploidy)

#예측 결과와 실제 데이터의 정확도 확인
sum(testPred==testData$ploidy)/length(testPred)*100


