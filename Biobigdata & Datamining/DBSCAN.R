rm(list = ls())

#install.packages("fpc")
library(fpc)
data(iris)
iris2 <- iris[-5]

ds <- dbscan(iris2, eps=0.42, MinPts=5)

#compare clusters with original class labels
table(ds$cluster, iris$Species) #행, 렬 로 테이블 출력
#원래 cluster 3개나와야하는데 오류있을 것것

 
plotcluster(iris2, ds$cluster)


