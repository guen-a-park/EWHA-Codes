rm(list = ls())

ts_data <- matrix(rnorm(16*10), ncol=16)  # 정규분포를 따르는 데이터 생성
ts_data
m_ts <- apply(ts_data, 2, mean)
sd_ts <- apply(ts_data, 2, sd)

ts_data_new <- ts_data

# Data Standardization/ 문제에서 0~100까지로 표현하라고 하면 생략하기 
# rescales data to have a mean of 0 and a standard deviation of 1 
for (i in 1:16)
  ts_data_new[,i] <-(ts_data[,i] - m_ts[i]) / sd_ts[i]

str(ts_data_new)

head(ts_data_new)

apply(ts_data_new, 2, mean)
apply(ts_data_new, 2, sd)

par(mfrow=c(4,4), mai=c(0.3, 0.6, 0.1,0.1))
#put multiple graphs in a single plot by setting some graphical parameters with the help of par() function
#par() function helps us in setting or inquiring about these parameters. 
# mfrow : specify the number of subplot
#A numerical vector of the form c(bottom, left, top, right) 
#which gives the margin size specified in inches.

for (i in 1:16) {
  plot(ts_data_new[,i], type="l")
  lines(ts_data_new[,1], col="red")
}

dist(rbind(ts_data_new[,1], ts_data_new[,2]), method = "euclidean")

euc_distances <- vector(length=15)

for (i in 2:16)
  euc_distances[i-1] <- dist(rbind(ts_data_new[,1], ts_data_new[,i]), method="euclidean")

euc_distances

par(mfrow=c(4,4), mai=c(0.3, 0.6, 0.1,0.1))
plot(ts_data_new[,1], type="l")
for(i in order(euc_distances)) {
  plot(ts_data_new[,i+1], type="l")
  lines(ts_data_new[,1], col="red")
}

#install.packages("dtw")
library(dtw)

dtw_alignment <- dtw(ts_data_new[,1], ts_data_new[,2], keep=T)
dtw_alignment$distance


dtwPlotTwoWay(dtw_alignment)

dtwPlotThreeWay(dtw_alignment)


dtw_distances <- vector(length=15)

for (i in 2:16){
  dtw_alignment <- dtw(ts_data_new[,1], ts_data_new[,i], keep=T)
  dtw_distances[i-1] <- dtw_alignment$distance
}

dtw_distances

euc_distances

par(mfrow=c(4,4), mai=c(0.3, 0.6, 0.1,0.1))

plot(ts_data_new[,1], type="l")
for(i in order(dtw_distances)) {
  plot(ts_data_new[,i+1], type="l")
  lines(ts_data_new[,1], col="red")
}