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

# prepare the data (see bootstrap_rsf_model.R for step by step explanations)
create_rsf_mod <- function(ds){
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
                   forest = T, seed= 5)
  return(rsf.out)
}

# Get top 10 minimal depth score and variables for each data group
rsf_1 <- create_rsf_mod(1)
md_1 <- max.subtree(rsf_1)
md_top10_1 <- data.frame(sort(md_1$order[, 1])[1:10])
colnames(md_top10_1) <- c("Minimal_depth")
md_top10_1 <- tibble::rownames_to_column(md_top10_1, "Covariate")
md_top10_1$dg <- rep("Full data")

rsf_2 <- create_rsf_mod(2)
md_2 <- max.subtree(rsf_2)
md_top10_2 <- data.frame(sort(md_2$order[, 1])[1:10])
colnames(md_top10_2) <- c("Minimal_depth")
md_top10_2 <- tibble::rownames_to_column(md_top10_2, "Covariate")
md_top10_2$dg <- rep("No bdgrp1 data")


rsf_3 <- create_rsf_mod(3)
md_3 <- max.subtree(rsf_3)
sorted_miRNA <- sort(md_3$order[, 1])
md_top10_3 <- data.frame(sort(md_3$order[, 1])[1:10])
colnames(md_top10_3) <- c("Minimal_depth")
md_top10_3 <- tibble::rownames_to_column(md_top10_3, "Covariate")
md_top10_3$dg <- rep("Complete metadata")

md_top10_plot <- rbind(md_top10_1, md_top10_2, md_top10_3)
md_top10_plot$dg <- as.factor(md_top10_plot$dg)


# Fix for appropriate ordering in the plot
scale_x_reordered <- function(..., sep = "___") {
  reg <- paste0(sep, ".+$")
  ggplot2::scale_x_discrete(labels = function(x) gsub(reg, "", x), ...)
}

reorder_within <- function(x, by, within, fun = mean, sep = "___", ...) {
  new_x <- paste(x, within, sep = sep)
  stats::reorder(new_x, by, FUN = fun, decreasing = TRUE)
}

# Make bar plot of top 10 impactful variables by data group
md_plot <- ggplot(md_top10_plot, aes(reorder_within(Covariate, Minimal_depth, dg), Minimal_depth)) + geom_bar(colour="black", fill="blue",stat="identity", width=0.4) +
  scale_x_reordered() +
  xlab("Covariate") + ylab("Minimal Depth") + labs(title="10 most impactful covariates for each data group ranked by minimal depth") + coord_flip()+facet_wrap(~dg, nrow=3, ncol=1, scales="free") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), axis.ticks.length.y = unit(.25, "cm"))

md_plot
