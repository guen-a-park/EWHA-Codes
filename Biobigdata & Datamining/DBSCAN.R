rm(list = ls())

#install.packages("fpc")
library(fpc)
data(iris)
iris2 <- iris[-5]

ds <- dbscan(iris2, eps=0.42, MinPts=5)

#compare clusters with original class labels
table(ds$cluster, iris$Species) #��, �� �� ���̺� ���
#���� cluster 3�����;��ϴµ� �������� �Ͱ�

 
plotcluster(iris2, ds$cluster)

