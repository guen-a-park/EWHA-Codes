rm(list = ls())

#install.packages("dtw")
library(dtw)
set.seed(seed=1234)

#난수 데이터 생성
ts_data_1 = c(sample(x=1:5, size=5, replace=FALSE))
ts_data_2 = c(sample(x=1:5, size=3, replace=FALSE))

#dtw distance 구하기
dtw_alignment <- dtw(ts_data_1, ts_data_2, keep=T, step.pattern = symmetric1)
dtw_alignment$distance

#plot 그리기
dtwPlotTwoWay(dtw_alignment)
dtwPlotThreeWay(dtw_alignment)
