#!/usr/bin/env Rscript

library(data.table)
library(randomForestSRC)
library(dplyr)
library(reshape2)
library(ggplot2)
library(timeROC)
library(survival)

set.seed(5)

args = commandArgs(trailingOnly=TRUE)
if (length(args) != 1) {
  stop("exactly 1 arguments must be supplied, integer from 1 to 3 to select dataset", call.=FALSE)
}

data_select = args[1]

# Select data group and import data
if (data_select == 1){
  data <- read.csv("path1")} else if(data_select == 2){
    data <- read.csv("path2")
  }else{
    data <- read.csv("path3")
  }

# select name for files
data_name <- ""
if (data_select == 1){
  data_name <- "name1"} else if(data_select == 2){
    data_name <- "name2"
  }else{
    data_name <- "name3"}

# Legcy for previous work, change from 50 years for controls to 10
data$tdato_diag_time_years <- replace(data$tdato_diag_time_years, data$tdato_diag_time_years == 50.0, 10.0)

# Reclassify 5 controls diagnosed with CRC cancer shortly after 10 years
names(data)[names(data) == "tdato_diag_time_years"] <- "time"
data <- within(data, condition["time" > 10.0 & "time" < 10.99] <- "CRC")
data <- within(data, time[condition == "C"] <- 10.0)


# Create dataset with only miRNAs and desiered confounders + time and event indicators
complete_data <- select(data, 3:372) 
complete_data$condition <- data$condition
complete_data$time <- data$time
complete_data$condition <- complete_data$condition %in% "CRC"


data[, "sex"] <- as.numeric(as.factor(data[, "sex"]))
data[, "smoking_status"] <- as.factor(data[, "smoking_status"])
data[, "physact_status"] <- as.factor(data[, "physact_status"])


covar <- c("sex", "alder_tdato")
if (data_select > 2){
  covar <- c("sex","alder_tdato", "smoking_status", "physact_status", "bmi")
}
complete_data[covar] <- data[covar]

# Get miRNAs chosen by stepwise selection and remove the remnaing for the dataset
chosen_variables <- read.csv(paste("output", data_name, sep="_"))
idx <- which.min(chosen_variables$rsf.err)
chosen_variables <- chosen_variables[idx:nrow(chosen_variables), ]
chosen_variables <- chosen_variables$rm.var

complete_data <- complete_data[, c(covar, chosen_variables, "condition", "time")]
complete_data <- na.omit(complete_data)

# Refit the model using only selected miRNAs with a new seed for B bootstraps
n <- 100
nsplit <- 10
ntree <- 1000
rsf.err.bs <- rep(0, n)
x <- c(0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 9.0)
rsf.auc <- matrix(0, n+1, length(x)) 
rsf.auc[1,] <- x

# miRNA + confounders
tuning <- tune(Surv(time, condition) ~ ., complete_data)
for (k in 1:n){
  set.seed(k)
  rsf.out <- rfsrc(Surv(time, condition) ~ ., data = complete_data, ntree = ntree, nsplit = nsplit, mtry=tuning$optimal[2], nodesize = tuning$optimal[1],
                   forest = T, seed=k)
  rsf.err.bs[k] <- rsf.out$err.rate[ntree]
  mortality_pred <- rsf.out$predicted.oob
  eval_times <- c(0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 9.0) 
  tv_auc <- timeROC(T =complete_data$time, delta =complete_data$condition, marker = mortality_pred, times=eval_times, cause = 1, iid=FALSE)
  rsf.auc[k+1, ] <- tv_auc$AUC

}

write.csv(rsf.err.bs, file=paste("rsf_rna_covars_err_bs", data_name, sep = "_"))
write.csv(rsf.auc, file=paste("rsf_rna_covars_auc", data_name, sep = "_"))

# just miRNA
complete_data <- complete_data[, c(chosen_variables, "condition", "time")]

n <- 100
nsplit <- 10
ntree <- 1000
rsf.err.bs <- rep(0, n)
x <- c(0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 9.0)
rsf.auc <- matrix(0, n+1, length(x)) 
rsf.auc[1,] <- x

tuning <- tune(Surv(time, condition) ~ ., complete_data)
for (k in 1:n){
  set.seed(k+100)
  rsf.out <- rfsrc(Surv(time, condition) ~ ., data = complete_data, ntree = ntree, nsplit = nsplit, mtry=tuning$optimal[2], nodesize = tuning$optimal[1],
                   forest = T, seed=k)
  rsf.err.bs[k] <- rsf.out$err.rate[ntree]
  mortality_pred <- rsf.out$predicted.oob
  eval_times <- c(0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 9.0) 
  tv_auc <- timeROC(T =complete_data$time, delta =complete_data$condition, marker = mortality_pred, times=eval_times, cause = 1, iid=FALSE)
  rsf.auc[k+1, ] <- tv_auc$AUC
  
}

write.csv(rsf.err.bs, file=paste("rsf_rna_err_bs", data_name, sep = "_"))
write.csv(rsf.auc, file=paste("rsf_rna_auc", data_name, sep = "_"))


# just confounders
complete_data <- data[c(covar, "condition", "time")]
complete_data$condition <- complete_data$condition %in% "CRC"
complete_data <- na.omit(complete_data)

n <- 100
nsplit <- 10
ntree <- 1000
rsf.err.bs <- rep(0, n)
x <- c(0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 9.0)
rsf.auc <- matrix(0, n+1, length(x)) 
rsf.auc[1,] <- x

tuning <- tune(Surv(time, condition) ~ ., complete_data)
for (k in 1:n){
  set.seed(k+200)
  rsf.out <- rfsrc(Surv(time, condition) ~ ., data = complete_data, ntree = ntree, nsplit = nsplit, mtry=tuning$optimal[2], nodesize = tuning$optimal[1],
                   forest = T, seed=k)
  rsf.err.bs[k] <- rsf.out$err.rate[ntree]
  mortality_pred <- rsf.out$predicted.oob
  eval_times <- c(0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 9.0) 
  tv_auc <- timeROC(T =complete_data$time, delta =complete_data$condition, marker = mortality_pred, times=eval_times, cause = 1, iid=FALSE)
  rsf.auc[k+1, ] <- tv_auc$AUC
}

write.csv(rsf.err.bs, file=paste("rsf_covars_err_bs", data_name, sep = "_"))
write.csv(rsf.auc, file=paste("rsf_covars_auc", data_name, sep = "_"))

