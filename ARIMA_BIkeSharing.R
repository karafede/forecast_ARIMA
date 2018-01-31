
# https://www.datascience.com/blog/introduction-to-forecasting-with-arima-in-r-learn-data-science-tutorials

library(ggplot2)
library(forecast)
library(tseries)

setwd("C:/Forecast_R")

daily_data = read.csv('day.csv', header=TRUE, stringsAsFactors=FALSE)

str(daily_data)

daily_data$Date = as.Date(daily_data$dteday)

# plot counts of bikes
ggplot(daily_data, aes(Date, cnt)) +
  theme_bw() +
  geom_line() + scale_x_date('month')  +
  ylab("Daily Bike Checkouts") +
  xlab("")


# make a time-series

count_ts = ts(daily_data[, c('cnt')])

# remove outliers
daily_data$clean_cnt = tsclean(count_ts)
str(daily_data)

ggplot() +
  theme_bw() +
  geom_line(data = daily_data, aes(x = Date, y = clean_cnt)) +
  ylab('Cleaned Bicycle Count')


# make movig averages (ma)
# weekly moving average
daily_data$cnt_ma = ma(daily_data$clean_cnt, order=7) # using the clean count with no outliers 
# monhly moving average
daily_data$cnt_ma30 = ma(daily_data$clean_cnt, order=30)  # (monthly)


ggplot() +
  theme_bw() +
  geom_line(data = daily_data, aes(x = Date, y = clean_cnt, colour = "Counts")) +
  geom_line(data = daily_data, aes(x = Date, y = cnt_ma,   colour = "Weekly Moving Average"), size=1)  +
  geom_line(data = daily_data, aes(x = Date, y = cnt_ma30, colour = "Monthly Moving Average"), size=1)  +
  ylab('Bicycle Count')



#################################################
###### Step 3: Decompose Your Data ##############
#################################################


# get seasonality, trend and cycle

str(daily_data)
count_ma = ts(na.omit(daily_data$cnt_ma), frequency=30)
# # make a deseasonalized time-series
decomp = stl(count_ma, s.window="periodic")
# remove seasonality
deseasonal_cnt <- seasadj(decomp)
plot(decomp)

# check if our time-series is stationary or not

adf.test(count_ma, alternative = "stationary")

# Augmented Dickey-Fuller Test

# data:  count_ma
# Dickey-Fuller = -0.2557, Lag order = 8, p-value = 0.99
# alternative hypothesis: stationary

# p-value is too high....the time-serie of count_ma is NOT STATIONARY


# visual tool in determining whether a time-series is stationary

# correlation between a series and its lags
Acf(count_ma, main='')
# or 
Pacf(count_ma, main='')

# R plots 95% significance boundaries as blue dotted lines. 
# There are significant autocorrelations with many lags in our bike series, as shown by the ACF plot below. 
# therefore statistical properties are not constant over time.


# Dickey-Fuller test

# Usually, non-stationary series can be corrected by a simple transformation such as differencing. 
# Differencing the series can help in removing its trend or cycles. The idea behind differencing is that, 
# if the original data series does not have constant properties over time, then the change from one period to another might. 
# The difference is calculated by subtracting one period's values from the previous period's values:

# Plotting the differenced series, we see an oscillating pattern around 0 with no visible strong trend. 
# This suggests that differencing of order 1 terms is sufficient and should be included in the model.

# use the deseasonalized time-series
count_d1 = diff(deseasonal_cnt, differences = 1)
plot(count_d1)
adf.test(count_d1, alternative = "stationary")

# Augmented Dickey-Fuller Test
# 
# data:  count_d1
# Dickey-Fuller = -9.9255, Lag order = 8, p-value = 0.01
# alternative hypothesis: stationary

# p-value is MUCH better now....the time-serie of count_ma is NOT STATIONARY


# repeat the tests to determine whether a time-series is stationary

Acf(count_d1, main='ACF for Differenced Series')
Pacf(count_d1, main='PACF for Differenced Series')

####  much better
# Auto correlation (ACF plot) is present at lag 1 and 2 and 7 

#################
# ARIMA model ###
#################

# ARIMA(p, d, q)
# p = number of lags
# d = degree of differencing
# q = determines the number of terms to include in the model

# use the deseasonalized time-series
auto.arima(deseasonal_cnt, seasonal=FALSE)

# Series: deseasonal_cnt
# ARIMA(1,1,1)
# 
# Coefficients:
#   ar1      ma1
# 0.5510  -0.2496
# s.e.  0.0751   0.0849
# 
# sigma^2 estimated as 26180:  log likelihood=-4708.91
# AIC=9423.82   AICc=9423.85   BIC=9437.57


fit<-auto.arima(deseasonal_cnt, seasonal=FALSE)
fcast <- forecast(fit)
plot(fcast)



# 


fit2 = arima(deseasonal_cnt, order=c(1,1,7))
fcast <- forecast(fit2)
plot(fcast)


# tsdisplay(residuals(fit2), lag.max=15, main='Seasonal Model Residuals')


# Call:
#   arima(x = deseasonal_cnt, order = c(1, 1, 7))
# 
# Coefficients:
#   ar1     ma1     ma2     ma3     ma4     ma5     ma6      ma7
# 0.2803  0.1465  0.1524  0.1263  0.1225  0.1291  0.1471  -0.8353
# s.e.  0.0478  0.0289  0.0266  0.0261  0.0263  0.0257  0.0265   0.0285


# now let's FORECAST with the output of the ARIMA model (fit2)

# forecast horizon h periods (30 days)
fcast <- forecast(fit2, h=30)
plot(fcast)


# let's include seasonamity in the AIMA model

fit_w_seasonality = auto.arima(deseasonal_cnt, seasonal=TRUE)
fit_w_seasonality


# Series: deseasonal_cnt 
# ARIMA(2,1,2)(1,0,0)[30]                    
# 
# Coefficients:
#   ar1      ar2      ma1     ma2    sar1
# 1.3644  -0.8027  -1.2903  0.9146  0.0100
# s.e.  0.0372   0.0347   0.0255  0.0202  0.0388
# 
# sigma^2 estimated as 24810:  log likelihood=-4688.59
# AIC=9389.17   AICc=9389.29   BIC=9416.68


seas_fcast <- forecast(fit_w_seasonality, h=1000)
plot(seas_fcast)



# end ###
#########