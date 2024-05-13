#!/usr/bin/env Rscript

# Original template provided in supplementary materials for:
#Dietrich, S. et al. (2016). ‘Random Survival Forest in practice: a method for modelling
#complex metabolomics data in time to event analysis’. In: International journal of
#epidemiology 45.5, pp. 1406–1420.

# Modified by Emil Jettli, 15.05.2024

library(data.table)
library(randomForestSRC)
library(dplyr)
library(reshape2)
library(ggplot2)
library(timeROC)
library(survival)

set.seed(5)

args = commandArgs(trailingOnly=TRUE)
if (length(args) != 4) {
  stop("exactly 4 arguments must be supplied", call.=FALSE)
}

data_select <- args[1]
full_set <-args[4]
if (data_select == 1 & full_set == "false"){
  data <- read.csv("path1")} else if(data_select == 2 & full_set == "false"){
    data <- read.csv("path2")
  }else if(data_select > 2 & full_set == "false"){
    data <- read.csv("path3")
  }

if (data_select == 1 & full_set == "true"){
  data <- read.csv("path4")} else if(data_select == 2 & full_set == "true"){
    data <- read.csv("path5")
  }else if(data_select > 2 & full_set == "true"){
    data <- read.csv("path6")
  }


data_name <- ""
if (data_select == 1 & full_set == "false"){
  data_name <- "name1"} else if(data_select == 2 & full_set == "false"){
    data_name <- "name2"
  }else if(data_select > 2 & full_set == "false"){
    data_name <- "name3"}

if (data_select == 1 & full_set == "true"){
  data_name <- "name4"} else if(data_select == 2 & full_set == "true"){
    data_name <- "name5"
  }else if(data_select > 2 & full_set == "true"){
    data_name <- "name6"}

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
if(args[2] != "none"){
meta_select <- args[2]
sub_select <- args[3]
if(meta_select == "metastasis"){
complete_data <- complete_data[complete_data[, meta_select] %in% c(sub_select,"Control"),]}else if(meta_select == "location"){
complete_data <- complete_data[complete_data[, meta_select] %in% c(sub_select,"control"),]}
}
complete_data <- na.omit(complete_data)

if(args[2] != "none"){
complete_data <- complete_data[, !(names(complete_data) %in% c(meta_select))]
}

count <- ncol(complete_data)- 2 - length(covar) # remove event time, censoring and non miRNA covariate columns from count
rsf.err <- rep(0,count)				# list to save error rates
rsf.err.auc <- rep(0, count)                     # list to save auc scores
rm.var <- rep(0,count)			# list to save removed variables
ntree <- 1000					# number of bootstrap samples
nsplit <- 10					# number of node splits
tuning <- tune(Surv(time, condition) ~ ., complete_data)
for(k in 1 : count){
  # grow the RSF
  if (k %% 25 == 0){
  tuning <- tune(Surv(time, condition) ~ ., complete_data)
  }
  rsf.out <- rfsrc(Surv(time, condition) ~ ., data = complete_data, ntree = ntree, nsplit = nsplit, mtry=tuning$optimal[2], nodesize = tuning$optimal[1],
                   forest = T, seed = k)
  # save error rate of current RSF
  mortality_pred <- rsf.out$predicted.oob
  
  eval_times <- c(9.9)
  tv_auc <- timeROC(T =complete_data$time, delta =complete_data$condition, marker = mortality_pred, times=eval_times, cause = 1, iid=FALSE)
  rsf.err.auc[k] <- tv_auc$AUC[2] 
  rsf.err[k] <- rsf.out$err.rate[ntree]

  # grow pilot tree for feature weights
  xvar.used <- rfsrc(Surv(time, condition) ~ ., data=complete_data, ntree=ntree, 
                     nodedepth = 6, perf.type="none",
                     var.used="all.trees", mtry = Inf, nsplit = 100)$var.used
  
  ## calculate minimal depth with supervision
  ## use number of times variable splits to guide random feature selection
  os <- rfsrc(Surv(time, condition) ~ ., data=complete_data, ntree=ntree, 
              xvar.wt = xvar.used, mtry = 2*sqrt(count + length(covar)), nsplit = nsplit, 
              importance= FALSE)
  
  # get list of variables ordered by their minimal depth values
  v.max <- max.subtree(os)
  d <- sort(round(v.max$order[, 1], 3))
  #-------------------------------------------------------------------
  # get the miRNA with worst minimal depth value, if a covariate has
  # the worst minimal depth value than ignore it and consider only miRNAś
  i <- 1
  m <- 0
  while(i > 0){ 
    outvar<-names(d[length(d)-m])
    print(k)
    print(outvar)
    if(length(outvar) == 0){
      outvar <- "fake_var"
    }
    # if outvar is a covariate then look next variable
    i <- length(i <- grep(outvar, covar))
    m <- m + 1
  } 
  #-------------------------------------------------------------------
  # save name of removed variable in a list
  rm.var[k]<-outvar
  # delete metabolite with worst minDepth from dataset
  complete_data<-complete_data[,!(names(complete_data) %in% outvar)]
  # remove old RSF to ensure free working memory 
  rm(rsf.out)
  rm(os)
  rm(xvar.used)
}

output<-cbind(rm.var,rsf.err, rsf.err.auc)

# check the output to find the set of miRNAs with the lowest RSF prediction error rate
if(args[2] != "none"){
write.csv(output, file=paste("output", data_name, sub_select, sep="_"))
}else{
  write.csv(output, file=paste("output", data_name, sep="_"))
}
