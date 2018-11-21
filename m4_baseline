# install.packages("devtools")
# devtools::install_github("carlanetto/M4comp2018")
# devtools::install_github("robjhyndman/M4metalearning")
# devtools::install_github("pmontman/tsfeatures")
# devtools::install_github("pmontman/customxgboost")
library(forecast)

## basic example code
library(M4comp2018)
data(M4)
names(M4[[1]])
#> [1] "st"     "x"      "n"      "type"   "h"      "period" "xx"
#extract yearly series
yearly_M4 <- Filter(function(l) l$period == "Yearly", M4)
#plot one of the series, in red the future data
#in black, the hitorical data
library(ggplot2)
library(forecast)
plot(ts(c(M4[[40773]]$x, M4[[40773]]$xx),
        start=start(M4[[40773]]$x), frequency = frequency(M4[[40773]]$x)),
     col="red", type="l", ylab="")
lines(M4[[40773]]$x, col="black")

#read the help file for documentation
# ?M4comp2018

# from https://github.com/robjhyndman/M4metalearning/blob/master/docs/metalearning_example.md
library(M4metalearning)
library(M4comp2018)
set.seed(31-05-2018)
# we start by creating the training and test subsets
indices <- sample(length(M4))
# sample only 15 series for estimation
M4_train <- M4[ indices[1:15]]
M4_test <- M4[indices[16:25]]
# we create the temporal holdout version of the training and test sets
M4_train <- temp_holdout(M4_train)
M4_test <- temp_holdout(M4_test)

# this will take time
M4_train <- calc_forecasts(M4_train, c("auto_arima_forec", 'ets_forec', 'tbats_forec'), n.cores=4)
# once we have the forecasts, we can calculate the errors
M4_train2 <- calc_errors(M4_train)
