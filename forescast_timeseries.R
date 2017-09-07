
# load a time series
data(AirPassengers)

str(AirPassengers)
class(AirPassengers)
frequency(AirPassengers)
summary(AirPassengers)



plot(AirPassengers)
abline(reg=lm(AirPassengers~time(AirPassengers)))
cycle(AirPassengers)

plot(aggregate(AirPassengers,FUN=mean))
boxplot(AirPassengers~cycle(AirPassengers))

(fit <- arima(log(AirPassengers), c(0, 1, 1),seasonal = list(order = c(0, 1, 1), period = 12)))
pred <- predict(fit, n.ahead = 10*12)
ts.plot(AirPassengers,2.718^pred$pred, log = "y", lty = c(1,3))

######################################################
# Forecasting time series with neural networks in R ##
######################################################

library(forecast)


setwd("D:/Forecast_R")

daily_data = read.csv('day.csv', header=TRUE, stringsAsFactors=FALSE)

str(daily_data)

daily_data$Date = as.Date(daily_data$dteday)
count_ts = ts(daily_data[, c('cnt')])

# remove outliers
daily_data$clean_cnt = tsclean(count_ts)
str(daily_data)


data <- as.data.frame (daily_data$clean_cnt)
data <-as.data.frame(AirPassengers)
plot(data)


y=auto.arima(AirPassengers)
plot(forecast(y,h=30))
points(1:length(AirPassengers),fitted(y),type="l",col="green")

