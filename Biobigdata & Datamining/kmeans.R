rm(list = ls())

data(wine, package="rattle.data")
head(wine)
=
#데이터 정규화는 변숫값의 분포를 표준화하는 것을 의미 
#표준화는 변수에서 데이터의 평균을 빼거나 
#변수를 전체 데이터의 표준 편차로 나누는 작업을 포함 
#변숫값의 평균이 0이 되고 값의 퍼짐 정도(분포) 또한 일정해진다.
df <- scale(wine[-1])  #without the first column

#군집 수에 따른 집단-내 제곱합(within-groups sum of squares)의 그래프
#data는 수치형의 자료이며, nc는 고려할 군집의 최대 수, seed는 난수발생 초기값
wssplot <- function(data, nc=15, seed=1234){ 
  wss <- (nrow(data)-1)*sum(apply(data,2,var)) 
   for (i in 2:nc){
                set.seed(seed)
                wss[i] <- sum(kmeans(data, centers=i)$withinss)
  } 
  plot(1:nc, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")
}

#군집 수를 정하기 위해
wssplot(df)
#군집 수 3에서 오차제곱합이 크게 감소되었음을 확인

#군집의 수(k)를 3으로 하여 kmeans()를 수행한 결과
#각 군집의 크기와 중심값을 보여준다. 
#군집 결과의 시각화는 plot() 함수를 이용
#nstart= generate 25 initial random centroids 

set.seed(1234)
fit.km <- kmeans(df, 3, nstart=25)
fit.km$size 

fit.km$centers

# 군집결과의  시각화
plot(df, col=fit.km$cluster)

#각 군집의 중심점추가
points(fit.km$center, col=1:2, pch=8, cex=1.5)

# 군집결과의  요약: (표준화된  자료에  대한)군집결과를 원자료의 단위로 전환
aggregate(wine[-1], by=list(cluster=fit.km$cluster), mean) 

# 군집결과의  성능: 정오분류표
ct.km <- table(wine$Type, fit.km$cluster)
ct.km
