#!/usr/bin/env Rscript

library(data.table)
library(randomForestSRC)
library(dplyr)
library(reshape2)
library(ggplot2)
library(gridExtra)
library(timeROC)
library(survival)
library(egg)
library(tidyr)

args = commandArgs(trailingOnly=TRUE)
if (length(args) != 1) {
  stop("exactly 1 argument must be supplied, use test set: T or F", call.=FALSE)
}

set.seed(5)
# Indicator to use holdout test set for evaluation
test_set <- args[1] == "T"

# prepare the data (see bootstrap_rsf_model.R for step by step explanations)
get_data <- function(ds, var_select){
  
  data <- read.csv("path3")
  data_name <- "name3"
  
  data$tdato_diag_time_years <- replace(data$tdato_diag_time_years, data$tdato_diag_time_years == 50.0, 10.0)
  
  
  names(data)[names(data) == "tdato_diag_time_years"] <- "time"
  data <- within(data, condition["tdato_diag_time_years" > 10.0 & "tdato_diag_time_years" < 10.99] <- "CRC")
  data <- within(data, time[condition == "C"] <- 10.0)

  
  complete_data <- select(data, 3:372)
  complete_data$condition <- data$condition
  complete_data$time <- data$time
  complete_data$condition <- complete_data$condition %in% "CRC"
  
  
  data[, "sex"] <- as.factor(data[, "sex"])
  data[, "smoking_status"] <- as.factor(data[, "smoking_status"])
  data[, "physact_status"] <- as.factor(data[, "physact_status"])
  

  covar <- c("sex", "alder_tdato", "smoking_status", "bmi", "physact_status")
  complete_data[covar] <- data[covar]
  s <- NULL
  # Create cancer specific subset
  if(ds == 1 & var_select == 1){
    complete_data$stage <- data$metastasis
    complete_data <- complete_data[complete_data$stage %in% c("Localized", "Control"), ]
    complete_data <- complete_data[, !(names(complete_data) %in% c("stage"))]
    s <- "localized"
  }else if(ds == 1 & var_select == 2){
    complete_data$stage <- data$metastasis
    complete_data <- complete_data[complete_data$stage %in% c("Regional", "Control"), ]
    complete_data <- complete_data[, !(names(complete_data) %in% c("stage"))]
    s <- "Regional"
  }else if(ds == 1 & var_select == 3){
    complete_data$stage <- data$metastasis
    complete_data <- complete_data[complete_data$stage %in% c("Distant", "Control"), ]
    complete_data <- complete_data[, !(names(complete_data) %in% c("stage"))]
    s <- "Distant"
  }else if(ds == 2 && var_select == 1){
    complete_data$location <- data$location
    complete_data <- complete_data[complete_data$location %in% c("distal", "control"), ]
    complete_data <- complete_data[, !(names(complete_data) %in% c("loacation"))]
    s <- "distal"
  }else{
    complete_data$location <- data$location
    complete_data <- complete_data[complete_data$location %in% c("proximal", "unspecified", "control"), ]
    complete_data <- complete_data[, !(names(complete_data) %in% c("loacation"))]
    s <- "proximal"
  }

  chosen_variables <- read.csv(paste("output", data_name, s, sep="_"))
  if(ds == 1 & var_select == 1){
    idx <- which.min(1-chosen_variables$rsf.err.auc)
  }else{
    idx <- which.min(chosen_variables$rsf.err) 
  }
  chosen_variables <- chosen_variables[idx:nrow(chosen_variables), ]
  chosen_variables <- chosen_variables$rm.var

  complete_data <- complete_data[, c(covar, chosen_variables, "condition", "time")]
  complete_data <- na.omit(complete_data)
  
  return(complete_data)
}

# create function to prepare holdout test data
get_test_data <- function(ds,var_select){
  
  data_test <- read.csv("path4")
  data_name <- "name3"
  
  data_test$tdato_diag_time_years <- replace(data_test$tdato_diag_time_years, data_test$tdato_diag_time_years == 50.0, 10.0)
  
  names(data_test)[names(data_test) == "tdato_diag_time_years"] <- "time"
  data_test <- within(data_test, condition["tdato_diag_time_years" > 10.0 & "tdato_diag_time_years" < 10.99] <- "CRC")
  data_test <- within(data_test, time[condition == "C"] <- 10.0)
  
  complete_data_test <- select(data_test, 3:372)
  complete_data_test$condition <- data_test$condition
  complete_data_test$time <- data_test$time
  complete_data_test$condition <- complete_data_test$condition %in% "CRC"
  
  data_test[, "sex"] <- as.factor(data_test[, "sex"])
  data_test[, "smoking_status"] <- as.factor(data_test[, "smoking_status"])
  data_test[, "physact_status"] <- as.factor(data_test[, "physact_status"])
  
  covar <- c("sex", "alder_tdato", "smoking_status", "bmi", "physact_status")
  complete_data_test[covar] <- data_test[covar]
  s <- NULL
  if(ds == 1 & var_select == 1){
    complete_data_test$stage <- data_test$metastasis
    complete_data_test <- complete_data_test[complete_data_test$stage %in% c("Localized", "Control"), ]
    complete_data_test <- complete_data_test[, !(names(complete_data_test) %in% c("stage"))]
    s <- "localized"
  }else if(ds == 1 & var_select == 2){
    complete_data_test$stage <- data_test$metastasis
    complete_data_test <- complete_data_test[complete_data_test$stage %in% c("Regional", "Control"), ]
    complete_data_test <- complete_data_test[, !(names(complete_data_test) %in% c("stage"))]
    s <- "Regional"
  }else if(ds == 1 & var_select == 3){
    complete_data_test$stage <- data_test$metastasis
    complete_data_test <- complete_data_test[complete_data_test$stage %in% c("Distant", "Control"), ]
    complete_data_test <- complete_data_test[, !(names(complete_data_test) %in% c("stage"))]
    s <- "Distant"
  }else if(ds == 2 && var_select == 1){
    complete_data_test$location <- data_test$location
    complete_data_test <- complete_data_test[complete_data_test$location %in% c("distal", "control"), ]
    complete_data_test <- complete_data_test[, !(names(complete_data_test) %in% c("loacation"))]
    s <- "distal"
  }else{
    complete_data_test$location <- data_test$location
    complete_data_test <- complete_data_test[complete_data_test$location %in% c("proximal", "unspecified", "control"), ]
    complete_data_test <- complete_data_test[, !(names(complete_data_test) %in% c("loacation"))]
    s <- "proximal"
  }
  
  chosen_variables <- read.csv(paste("output", data_name, s, sep="_"))
  if(ds == 1 & var_select == 1){
    idx <- which.min(1-chosen_variables$rsf.err.auc)
  }else{
    idx <- which.min(chosen_variables$rsf.err) 
  }
  chosen_variables <- chosen_variables[idx:nrow(chosen_variables), ]
  chosen_variables <- chosen_variables$rm.var
  
  complete_data_test <- complete_data_test[, c(covar, chosen_variables, "condition", "time")]
  complete_data_test <- na.omit(complete_data_test)
  
  return(complete_data_test)
}

# placement for results
eval_times <- c(0.5, 1, 1.5, 2, 3, 5, 7, 9)
plotting_auc <- data.frame()
tp_df <- matrix(nrow =5, ncol=length(eval_times))
tp_se_df <- matrix(nrow=5, ncol=length(eval_times))
fp_df <- matrix(nrow= 5, ncol=length(eval_times))
fp_se_df <- matrix(nrow= 5, ncol=length(eval_times))
ppv_df <- matrix(nrow=5, ncol=length(eval_times))
ppv_se_df <- matrix(nrow=5, ncol=length(eval_times))
npv_df <- matrix(nrow=5, ncol=length(eval_times))
npv_se_df <- matrix(nrow=5, ncol=length(eval_times))

# Overly complicated loop to select one cancer specific subset at a time
i <- 0
for( l in 1:2){
  for(k in 1:3){
    if(l == 2 & k > 2){
      break
    }
    i <- i + 1
    complete_data <- get_data(l, k)
    complete_data_test <- get_test_data(l, k)
    n <- 1
    nsplit <- 10
    ntree <- 1000
    
    tuning <- tune(Surv(time, condition) ~ ., complete_data)
    
    # create cancer specific model using chosen miRNAs
    rsf.out <- rfsrc(Surv(time, condition) ~ ., data = complete_data, ntree = ntree, nsplit = nsplit, mtry=tuning$optimal[2], nodesize = tuning$optimal[1],
                     forest = T, seed = l+k)
    
    # create appropriate name
    if(l == 1 & k == 1){data_set <- "Stage: Localized"}else if(l == 1 & k == 2){data_set <-"Stage: Regional"}else if(l == 1 & k == 3){data_set <- "Stage: Distant"}else if(l == 2 & k == 1){
      data_set <- "Location: Distal"}else{
        data_set <- "Location: Proximal + Unspecified"}
    
    # Get predictions
    if(test_set == T){
      mortality_pred <- predict.rfsrc(rsf.out, complete_data_test, outcome = "test")$predicted.oob
      T_marker <- complete_data_test$time
      delta_marker <- complete_data_test$condition
    }else{
      mortality_pred <- rsf.out$predicted.oob
      T_marker <- complete_data$time
      delta_marker <- complete_data$condition
    }
    
    # Calculate time-varying AUC
    tv_auc <- timeROC(T =T_marker, delta =delta_marker, marker = mortality_pred , times=eval_times, cause = 1, iid=TRUE)
    
    # get and save C-index
    write.csv(get.cindex(delta_marker, T_marker, mortality_pred), file=paste(paste(data_set, "c_index_indv",  sep="_"), "csv", sep = "."))
    
    # Calculate time-varying sensitivity, specificity, ppv, npv:
    mortality_quantiles <- quantile(mortality_pred, probs = seq(0, 1, by =.1))
    tp_list <- matrix(0, length(mortality_quantiles), length(eval_times))
    fp_list <- matrix(0, length(mortality_quantiles), length(eval_times))
    ppv_list <- matrix(0, length(mortality_quantiles), length(eval_times))
    npv_list <- matrix(0, length(mortality_quantiles), length(eval_times))
    for( j in 1:length(mortality_quantiles)){ 
      sen <- SeSpPPVNPV(cutpoint=mortality_quantiles[j], T = T_marker, delta = delta_marker, marker = mortality_pred, cause = 1, times = eval_times, iid=TRUE)
      tp_list[j,] <- sen$TP 
      fp_list[j,] <- sen$FP
      ppv_list[j,] <- sen$PPV
      npv_list[j, ] <- sen$NPV
    }
    tp_list <- data.frame(tp_list)
    fp_list <- data.frame(fp_list)
    
    # Maximize Youden's J statistic and select performance metrics accordingly 
    J_stat <- tp_list - fp_list
    cp <- floor(mean(apply(J_stat, 2, function(x) c(max = max(x), ind = which.max(x)))[2,]))
    sen <- SeSpPPVNPV(cutpoint=mortality_quantiles[cp], T = T_marker, delta = delta_marker, marker = mortality_pred, cause = 1, times = eval_times, iid=TRUE)
    
    tp_df[i, ] <- unname(sen$TP)
    tp_se_df[i, ] <- unname(sen$inference$vect_se_Se)
    fp_df[i, ] <- unname(sen$FP)
    fp_se_df[i, ] <- unname(sen$inference$vect_se_Sp1)
    ppv_df[i, ] <- unname(sen$PPV)
    ppv_se_df[i, ] <- unname(sen$inference$vect_se_PPV)
    npv_df[i, ] <- unname(sen$NPV)
    npv_se_df[i, ] <- unname(sen$inference$vect_se_NPV2)
    
    # Get confidence bands data for plotting
    confidence_bands <- data.frame(unname(confint(tv_auc, level = 0.95, n.sim=3000)$CB_AUC))*0.01
    auc <- data.frame(unname(tv_auc$AUC))
    auc_to_plot <- cbind(confidence_bands, tv_auc$AUC)
    rownames(auc_to_plot) <- NULL
    auc_to_plot$data <- rep(data_set, nrow(auc_to_plot))
    auc_to_plot$t <- eval_times
    colnames(auc_to_plot) <- c("p05", "p95", "AUC", "data", "time")
    plotting_auc <- rbind(plotting_auc, auc_to_plot)
  }
}

title_str <- "Time varying AUC with 95% confidence bands for staging and location\n specific subsets using miRNA + confounders"
if(test_set == T){
  title_str <- "Time varying AUC for holdout test set with 95% confidence bands for staging\n and location specific subsets using miRNA + confounders"
}

# Create AUC plots
ggplot(plotting_auc, aes(x = time, y = AUC, groups=data)) +
  geom_line(col='blue') +
  geom_ribbon(aes(x = time, ymin = p05, ymax = p95), fill = "blue", alpha = 0.2) +
  geom_hline(yintercept = 0.5, size=0.2, color="blue", linetype=2) +
  labs(x = "Time to diagnosis", y = "AUC") +
  ggtitle(title_str) +
  coord_cartesian(ylim = c(0.3, 0.9)) + scale_y_continuous(n.breaks=10) + 
  facet_wrap(~data, nrow=3, ncol=2) + theme(axis.text.x = element_text(size=10), 
                                            axis.text.y = element_text(size=10),
                                            strip.text = element_text(size=10))

# Prepare remaining metrics and save as dataframe csv's
tp_df = data.frame(tp_df)
tp_se_df = data.frame(tp_se_df)
fp_df = data.frame(fp_df)
fp_se_df = data.frame(fp_se_df)

ppv_df = data.frame(ppv_df)
ppv_se_df = data.frame(ppv_se_df)
npv_df = data.frame(npv_df)
npv_se_df = data.frame(npv_se_df)

colnames(tp_df) <- eval_times
colnames(tp_se_df) <- eval_times
colnames(fp_df) <- eval_times
colnames(fp_se_df) <- eval_times
colnames(ppv_df) <- eval_times
colnames(ppv_se_df) <- eval_times
colnames(npv_df) <- eval_times
colnames(npv_se_df) <- eval_times

tp <- tp_df
tp<- paste0(round(as.matrix(tp_df), 2), " (", 
                 +                   round(as.matrix(tp_se_df), 2), ")" )
tp <- matrix(tp, nrow=5, ncol=8)
tp <- data.frame(tp)


fp <- fp_df
fp <- paste0(round(as.matrix(fp_df), 2), " (", 
                 +                   round(as.matrix(fp_se_df), 2), ")" )
fp <- matrix(fp, nrow=5, ncol=8)
fp <- data.frame(fp)

ppv <- ppv_df
ppv <- paste0(round(as.matrix(ppv_df), 2), " (", 
                  +                   round(as.matrix(ppv_se_df), 2), ")" )
ppv <- matrix(ppv, nrow=5, ncol=8)
ppv <- data.frame(ppv)

npv <- npv_df
npv <- paste0(round(as.matrix(npv_df), 2), " (", 
                  +                   round(as.matrix(npv_se_df), 2), ")" )
npv <- matrix(npv, nrow=5, ncol=8)
npv <- data.frame(npv)

rownames(tp) <- c("Localized", "Regional", "Distant", "Distal", "Proximal + Unspecified")
rownames(fp) <- c("Localized", "Regional", "Distant", "Distal", "Proximal + Unspecified")
rownames(ppv) <- c("Localized", "Regional", "Distant", "Distal", "Proximal + Unspecified")
rownames(npv) <- c("Localized", "Regional", "Distant", "Distal", "Proximal + Unspecified")

tp_str <- "tp_df_cancer.csv"
fp_str <- "fp_df_cancer.csv"
ppv_str <- "ppv_df_cancer.csv"
npv_str <- "npv_df_cancer.csv"

if(test_set == T){
  tp_str <- "tp_df_cancer_test.csv"
  fp_str <- "fp_df_cancer_test.csv"
  ppv_str <- "ppv_df_cancer_test.csv"
  npv_str <- "npv_df_cancer_test.csv"
}

write.csv(tp, tp_str)
write.csv(fp, fp_str)
write.csv(ppv, ppv_str)
write.csv(npv, npv_str)
