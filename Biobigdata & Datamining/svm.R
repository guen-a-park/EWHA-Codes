rm(list = ls())

library("e1071")
data(iris)

svm.model <- svm(Species ~ . , data = iris,
                   type = "C-classification", kernel = "radial", cost = 10, gamma = 0.1)
summary(svm.model)

#결과를 시각화
plot(svm.model, iris, Petal.Width ~ Petal.Length, slice = list(Sepal.Width = 3, Sepal.Length = 4))

#slice 옵션:
#반응에 대한 예측변수의 효과를 시각화할 때 
#어떤 다른 예측변수가 일정하게 유지될지(즉, 고정 값에서) 지정
#Sepal.Width 및 Sepal.Length를 지정된 값에서 일정하게 유지하면서 
#예측 변수 Petal.Length 및 Petal.Width가 응답에 미치는 영향을 시각화

plot(svm.model, iris, Sepal.Width ~ Sepal.Length,  slice = list(Petal.Width = 2.5, Petal.Length = 3))
#그림에서  x: 서포트벡터, o:데이터  점을  나타냄

#Classificatioin with test set
pred <- predict(svm.model, iris, decision.values = TRUE)

#분류결과 plotting
pred
plot(pred)


#분류된 데이터를 실제 값과 비교
acc <- table(pred, iris$Species)
 

#예측 결과와 실제 데이터의 정확도 확인
sum(pred==iris$Species)/length(pred)*100

#모형의 정확도 확인 
classAgreement(acc)  #diag - Percentage of data points in the main diagonal of the table


  