rm(list = ls())
data(USArrests)
str(USArrests)

# 계층적  군집  수행
d <- dist(USArrests, method="euclidean")  # 주들  간의  유클리드  거리
fit <- hclust(d, method="ave")

#dist()는 거리(또는 비유사성)행렬을 제공하는 함수
#method= 옵션을 통해 다양한 방식으로 거리를 정의
#method= 옵션에는 "euclidean", "manhattan", "minkowski" 등

#hclust() 함수는 계층적 군집분석을 수행하는 함수
#method= 옵션을 통해 병합(또는 연결) 방법을 지정
#method= 옵션에는 "ward", "single", "complete", "average", "centroid" 등



# 군집결과의  시각화: plot() 함수를 통해 덴드로그램으로 시각화 
plot(fit)


rect.hclust(fit, k=6, border="red")
#plot() 함수를 이용해 그린 덴드로그램은 rect.hclust() 함수를 이용하여 
#그룹을 사각형으로 구분지어 표현. 


rect.hclust(fit, h = 50, which = c(2,3), border = 5:6)
# 높이(h) 50에서  cut 수행, 2와  3번에  사각형  추가, 테두리  색상(border) 지정

#rect.hclust() 함수는 그룹수(k)를 이용하여 그룹을 시각화 할 뿐 아니라,
# tree의 높이(h)와 위치(which)를 이용하여 그룹의 전체 또는 일부를 표시.

###############################################################################

library(cluster)
# stand=TRUE는  비유사성(거리)의  계산  전에  표준화를  수행함. 
#입력 자료가  비유사성  행렬인  경우에는  무시됨
(agn1 <- agnes(USArrests, metric="manhattan", stand=TRUE))
plot(agn1)

# diss=TRUE는  입력  자료가  비유사성  행렬  또는  dist 객체인  경우에는  디폴트임
agn2 <- agnes(daisy(USArrests), diss=TRUE, method="complete")
plot(agn2)

