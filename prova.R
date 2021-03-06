

library(ggplot2)
library(forecast)
library(tseries)

value <- c(1.2, 1.7, 1.6, 1.2, 1.6, 1.3, 1.5, 1.9, 5.4, 4.2, 5.5, 6, 5.6, 
           6.2, 6.8, 7.1, 7.1, 5.8, 0, 5.2, 4.6, 3.6, 3, 3.8, 3.1, 3.4, 
           2, 3.1, 3.2, 1.6, 0.6, 3.3, 4.9, 6.5, 5.3, 3.5, 5.3, 7.2, 7.4, 
           7.3, 7.2, 4, 6.1, 4.3, 4, 2.4, 0.4, 2.4)

sensor<-ts(value,frequency=24)  # 24 hours
fit <- auto.arima(sensor)
LH.pred<-predict(fit,n.ahead=24)
plot(sensor,ylim=c(0,10),xlim=c(0,5),type="o", lwd="1")
lines(LH.pred$pred,col="red",type="o",lwd="1")
grid()


library(forecast)
value <- c(1.2,1.7,1.6, 1.2, 1.6, 1.3, 1.5, 1.9, 5.4, 4.2, 5.5, 6.0, 5.6, 6.2, 6.8, 7.1, 7.1, 5.8, 0.0, 5.2, 4.6, 3.6, 3.0, 3.8, 3.1, 3.4, 2.0, 3.1, 3.2, 1.6, 0.6, 3.3, 4.9, 6.5, 5.3, 3.5, 5.3, 7.2, 7.4, 7.3, 7.2, 4.0, 6.1, 4.3, 4.0, 2.4, 0.4, 2.4, 1.2,1.7,1.6, 1.2, 1.6, 1.3, 1.5, 1.9, 5.4, 4.2, 5.5, 6.0, 5.6, 6.2, 6.8, 7.1, 7.1, 5.8, 0.0, 5.2, 4.6, 3.6, 3.0, 3.8, 3.1, 3.4, 2.0, 3.1, 3.2, 1.6, 0.6, 3.3, 4.9, 6.5, 5.3, 3.5, 5.3, 7.2, 7.4, 7.3, 7.2, 4.0, 6.1, 4.3, 4.0, 2.4, 0.4, 2.4)
sensor <- ts(value,frequency=24) # consider adding a start so you get nicer labelling on your chart. 
fit <- auto.arima(sensor)
fcast <- forecast(fit)
plot(fcast)
grid()
fcast


library(forecast)
value <- c(1.2,1.7,1.6, 1.2, 1.6, 1.3, 1.5, 1.9, 5.4, 4.2, 5.5, 6.0, 5.6, 6.2, 6.8, 7.1, 7.1, 5.8, 0.0, 5.2, 4.6, 3.6, 3.0, 3.8, 3.1, 3.4, 2.0, 3.1, 3.2, 1.6, 0.6, 3.3, 4.9, 6.5, 5.3, 3.5, 5.3, 7.2, 7.4, 7.3, 7.2, 4.0, 6.1, 4.3, 4.0, 2.4, 0.4, 2.4, 1.2,1.7,1.6, 1.2, 1.6, 1.3, 1.5, 1.9, 5.4, 4.2, 5.5, 6.0, 5.6, 6.2, 6.8, 7.1, 7.1, 5.8, 0.0, 5.2, 4.6, 3.6, 3.0, 3.8, 3.1, 3.4, 2.0, 3.1, 3.2, 1.6, 0.6, 3.3, 4.9, 6.5, 5.3, 3.5, 5.3, 7.2, 7.4, 7.3, 7.2, 4.0, 6.1, 4.3, 4.0, 2.4, 0.4, 2.4)
sensor <- ts(value,frequency=24) # consider adding a start so you get nicer labelling on your chart. 
fit <- auto.arima(sensor)
fit2 = arima(sensor, order=c(4,1,9))
fcast <- forecast(fit2, h = 30)
plot(fcast)
grid()
fcast
