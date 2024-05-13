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
library(tibble)

set.seed(5)

# prep data, selected core set of miRNAs and prepare partial depedence plots
create_rsf_mod <- function(ds, data_set){
  data_select = ds
  
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
  data <- within(data, condition["tdato_diag_time_years" > 10.0 & "tdato_diag_time_years" < 10.99] <- "CRC")
  data <- within(data, time[condition == "C"] <- 10.0)
  
  complete_data <- select(data, 3:372) 
  complete_data$condition <- data$condition
  complete_data$time <- data$time
  complete_data$condition <- complete_data$condition %in% "CRC"
  
  
  data[, "sex"] <- as.factor(data[, "sex"])
  data[, "smoking_status"] <- as.factor(data[, "smoking_status"])
  data[, "physact_status"] <- as.factor(data[, "physact_status"])
  
  
  covar <- c("sex", "alder_tdato")
  if (data_select > 2){
    covar <- c("sex", "alder_tdato", "smoking_status", "bmi", "physact_status")
  }
  complete_data[covar] <- data[covar]
  
  chosen_variables <- read.csv(paste("output", data_name, sep="_"))
  idx <- which.min(chosen_variables$rsf.err)
  chosen_variables <- chosen_variables[idx:nrow(chosen_variables), ]
  chosen_variables <- chosen_variables$rm.var
  
  complete_data <- complete_data[, c(covar, chosen_variables, "condition", "time")]
  complete_data <- na.omit(complete_data)

  n <- 1
  nsplit <- 10
  ntree <- 1000
  rsf.err.bs <- rep(0, n)
  x <- c(0.5, 1, 1.5, 2, 3, 5, 7, 9)
  rsf.auc <- matrix(0, n+1, length(x)) 
  rsf.auc[1,] <- x
  
  tuning <- tune(Surv(time, condition) ~ ., complete_data)
  
  rsf.out <- rfsrc(Surv(time, condition) ~ ., data = complete_data, ntree = ntree, nsplit = nsplit, mtry=tuning$optimal[2], nodesize = tuning$optimal[1],
                     forest = T)
  
  # partial plots
  create_partial_plots <- function(i, data_var, rsf_mod){ 
    ## get partial effect of age on mortality
    partial.obj <- partial(rsf.out,
                           oob=TRUE,
                           partial.type = "mort",
                           partial.xvar = rsf.out$xvar.names[i],
                           partial.values = rsf.out$xvar[, i],
                           partial.time = rsf.out$time.interest)
    pdata <- get.partial.plot.data(partial.obj)
    
    x <- pdata$x
    yhat <- pdata$yhat
    cov <- rep(rsf.out$xvar.names[i], length(x))
    data_select <- rep(data_var, length(x))
    pdata <- data.frame(cbind(x, yhat, cov, data_select))
    pdata$x <- as.numeric(pdata$x)
    pdata$yhat <- as.numeric(pdata$yhat)
    
    return(pdata)
  }
  
  
  output_all <- read.csv("path7")
  
  all_miRNA <- output_all$rm.var
  
  idx <- which.min(output_all$rsf.err)
  output_all <- output_all[idx:nrow(output_all), ]
  output_all <- output_all$rm.var
  
  output_no_bdgrp1 <- read.csv("path8")
  idx <- which.min(output_no_bdgrp1$rsf.err)
  output_no_bdgrp1 <- output_no_bdgrp1[idx:nrow(output_no_bdgrp1), ]
  output_no_bdgrp1 <- output_no_bdgrp1$rm.var
  
  output_no_bdgrp1_no_NA <- read.csv("path9")
  idx <- which.min(output_no_bdgrp1_no_NA$rsf.err)
  output_no_bdgrp1_no_NA <- output_no_bdgrp1_no_NA[idx:nrow(output_no_bdgrp1_no_NA), ]
  output_no_bdgrp1_no_NA <- output_no_bdgrp1_no_NA$rm.var
  
  
  common_1_2 <- length(intersect(output_all, output_no_bdgrp1))
  common_1_3 <- length(intersect(output_all, output_no_bdgrp1_no_NA))
  common_2_3 <- length(intersect(output_no_bdgrp1, output_no_bdgrp1_no_NA))
  common_1_2_3 <- length(intersect(intersect(output_no_bdgrp1, output_all), output_no_bdgrp1_no_NA))

  # Output the results
  print(paste("Number of common elements between output_all and output_no_bdgrp1:", common_1_2))
  print(paste("Number of common elements between output_all and output_no_bdgrp1_no_NA:", common_1_3))
  print(paste("Number of common elements between output_no_bdgrp1 and output_no_bdgrp1_no_NA:", common_2_3))
  print(paste("Number of common elements between all outputs:", common_1_2_3))
  
  # as proportions
  prop_all_nobdgrp1 <- common_1_2 / length(output_all) 
  prop_nobdgrp1_all <- common_1_2 / length(output_no_bdgrp1)
  prop_all_noNA <-  common_1_3 / length(output_all)
  prop_noNA_all <-  common_1_3 / length(output_no_bdgrp1_no_NA)
  prop_nobdgrp1_noNA <- common_2_3 /  length(output_no_bdgrp1)
  prop_noNA_nobdgrp1 <- common_2_3 / length(output_no_bdgrp1_no_NA)
  print(paste("proportion of all to no_bdgrp1: ", prop_all_nobdgrp1, "proportion all to no_NA: ", prop_all_noNA))
  print(paste("proportion of no_bdgrp1 to all: ", prop_nobdgrp1_all, "proportion no_bdgrp1 to no_NA: ", prop_nobdgrp1_noNA))
  print(paste("proportion of no_NA to no_bdgrp1: ", prop_noNA_nobdgrp1, "proportion no_NA to all: ", prop_noNA_all))
  
  print("The 12 chosen miRNAs across all 3 conditions: ") 
  c1_2 <- intersect(output_all, output_no_bdgrp1)
  c1_3 <- intersect(output_all, output_no_bdgrp1_no_NA)
  c2_3 <- intersect(output_no_bdgrp1, output_no_bdgrp1_no_NA)
  c1_2_3 <- intersect(intersect(c1_2, c1_3), c2_3)
  print(c(c1_2_3))
  
  common_mirnas <- c1_2_3 
  
  # Get only core set for partial dependence plotting
  used_var_data <- complete_data[,c(covar, chosen_variables)]
  plotting_df <- create_partial_plots(1, data_set)

  plotting_df <- rbind(plotting_df, create_partial_plots(2, data_set))

  r <- 3
  if(data_set == "Complete metadata"){
    plotting_df <- rbind(plotting_df, create_partial_plots(3, data_set))
    plotting_df <- rbind(plotting_df, create_partial_plots(4, data_set))
    plotting_df <- rbind(plotting_df, create_partial_plots(5, data_set))
    r <- 6
  }
  
  for( i in r:ncol(used_var_data)){
    n <- rsf.out$xvar.names[i]
    if(n %in% common_mirnas){
      plotting_df <- rbind(plotting_df, create_partial_plots(i, data_set))

    }
  }
  return(plotting_df)
}

# create partial dependence plots
df_plotting <- create_rsf_mod(1, "Full")
df_plotting <- rbind(df_plotting, create_rsf_mod(2, "No bdgrp1"))
df_plotting <- rbind(df_plotting, create_rsf_mod(3, "Complete metadata"))

p <- ggplot(df_plotting, aes(x = x, y = yhat, color=data_select, group=data_select)) + geom_line() + facet_wrap(vars(cov), scales = 'free_x') +
              labs(x = "covariate value", y = "Adjusted mortality", title = "Partial dependence plots for mortality against core set of variables")
p

# Create predictions error plots
output_all <- read.csv("path4")
output_all$data <- rep("Full", nrow(output_all))
output_all <- output_all[,c("X","rsf.err", "data")]
output_no_bdgrp1 <- read.csv("path5")
output_no_bdgrp1$data <- rep("No bdgrp1", nrow(output_no_bdgrp1))
output_no_bdgrp1 <- output_no_bdgrp1[,c("X","rsf.err", "data")]
output_no_bdgrp1_no_NA <- read.csv("path6")
output_no_bdgrp1_no_NA$data <- rep("Complete metadata", nrow(output_no_bdgrp1_no_NA))
output_no_bdgrp1_no_NA <- output_no_bdgrp1_no_NA[,c("X","rsf.err", "data")]

max_it <- max(output_all$X)
output_all$X <- rep(max_it, nrow(output_all)) - output_all$X 
output_no_bdgrp1$X <- rep(max_it, nrow(output_no_bdgrp1)) - output_no_bdgrp1$X 
output_no_bdgrp1_no_NA$X <- rep(max_it, nrow(output_no_bdgrp1_no_NA)) - output_no_bdgrp1_no_NA$X 

output_df <- rbind(output_all, output_no_bdgrp1, output_no_bdgrp1_no_NA)

all_min <- output_all[which.min(output_all$rsf.err), ]$X
all_min_err <-output_all[which.min(output_all$rsf.err), ]$rsf.err

no_bdgrp1_min <- output_no_bdgrp1[which.min(output_no_bdgrp1$rsf.err),]$X
no_bdgrp1_min_err <-output_no_bdgrp1[which.min(output_no_bdgrp1$rsf.err), ]$rsf.err

no_NA_min <- output_no_bdgrp1_no_NA[which.min(output_no_bdgrp1_no_NA$rsf.err), ]$X
no_NA_min_err <-output_no_bdgrp1_no_NA[which.min(output_no_bdgrp1_no_NA$rsf.err), ]$rsf.err

ggplot(output_df, aes(x = X, y = rsf.err, color=data, groups=data)) + geom_line() + scale_x_reverse() +geom_point(aes(x=all_min, y=all_min_err), colour="green") + 
  geom_point(aes(x=no_bdgrp1_min, y=no_bdgrp1_min_err), colour="blue") + geom_point(aes(x=no_NA_min, y=no_NA_min_err), colour="red") + 
  labs(x="Remaining miRNA covariates", y = "Prediction err (1-C-index)", title = "Prediction error curves for stepwise rsf variable selection") +
  theme(legend.justification = c('bottom'), legend.key = element_rect(fill = "transparent"))

geom_vline(xintercept = all_min, linetype="dashed", color="green") +
  geom_vline(xintercept = no_bdgrp1_min, linetype="dashed", color="blue") + geom_vline(xintercept = no_NA_min, linetype="dashed", color="red") 