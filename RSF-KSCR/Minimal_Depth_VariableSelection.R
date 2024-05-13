#!/usr/bin/env Rscript

library(data.table)
library(randomForestSRC)
library(dplyr)
library(reshape2)
library(ggplot2)

args = commandArgs(trailingOnly=TRUE)
if (length(args) != 1) {
  stop("exactly 1 argument must be supplied, integer from 1 to 3 to select data group", call.=FALSE)
}

set.seed(5)

data_select = args[1]
if (data_select == 1){
  data <- read.csv("path1")} else if(data_select == 2){
    data <- read.csv("path2")
  }else{
    data <- read.csv("path3")
  }

data_name <- ""
if (data_select == 1){
  data_name <- "name1"} else if(data_select == 2){
    data_name <- "name2"
  }else{
    data_name <- "name3"}

data$tdato_diag_time_years <- replace(data$tdato_diag_time_years, data$tdato_diag_time_years == 50.0, 10.0)

names(data)[names(data) == "tdato_diag_time_years"] <- "time"
data <- within(data, condition["time" > 10.0 & "time" < 10.99] <- "CRC")
data <- within(data, time[condition == "C"] <- 10.0)
data$time <- data$time #* 52.177457

complete_data <- select(data, 3:372) 
complete_data$condition <- data$condition
complete_data$time <- data$time
complete_data$condition <- complete_data$condition %in% "CRC"

data[, "sex"] <- as.numeric(as.factor(data[, "sex"]))
data[, "smoking_status"] <- as.factor(data[, "smoking_status"])
data[, "physact_status"] <- as.factor(data[, "physact_status"])

covar <- c("sex", "alder_tdato")
if (data_select > 2){
  covar <- c("sex", "alder_tdato" ,"smoking_status", "physact_status", "bmi")
}
complete_data[covar] <- data[covar]
  
xvar.used <- rfsrc(Surv(time, condition) ~ ., data=complete_data, ntree=1000, nodedepth = 6, perf.type="none", 
                   var.used="all.trees", mtry = Inf, nsplit = 100)$var.used

## calculate minimal depth with supervision
## use number of times variable splits to guide random feature selection
os <- rfsrc(Surv(time, condition) ~ ., data=complete_data, ntree=1000, xvar.wt = xvar.used, mtry = 40, 
            importance= FALSE)

# get minimal depth scores
mst <- max.subtree(os, sub.order = TRUE)
mds <- mst$order[, 1]

# Use expected value as threshold, retaining only miRNAs with lower minimal depth
output <- mds[mds <= mst$threshold]
write.csv(output, file=paste("md_var_sel", data_name, sep="_"))
