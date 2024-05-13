library(data.table)
library(dplyr)
library(reshape2)
library(ggplot2)
library(gridExtra)
library(tidyr)
library(gtsummary)
library(smd)
set.seed(5)

# prepare the data (see bootstrap_rsf_model.R for step by step explanations)
get_data <- function(ds){
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

  complete_data <- within(complete_data, condition[condition == "CRC"] <- "Cases")
  complete_data <- within(complete_data, condition[condition == "C"] <- "Controls")
  
  data[, "sex"] <- as.factor(data[, "sex"])
  data[, "smoking_status"] <- as.factor(data[, "smoking_status"])
  data[, "physact_status"] <- as.factor(data[, "physact_status"])
  data[, "bd_grp"] <- as.factor(data[, "bd_grp"])
  
  
  covar <- c("sex", "alder_tdato", "smoking_status", "bmi", "physact_status", "bd_grp")
  if (data_select > 2){
    covar <- c("sex", "alder_tdato", "smoking_status", "bmi", "physact_status", "bd_grp")
  }
  complete_data[covar] <- data[covar]

  chosen_variables <- read.csv(paste("output", data_name, sep="_"))
  idx <- which.min(chosen_variables$rsf.err)
  chosen_variables <- chosen_variables[idx:nrow(chosen_variables), ]
  chosen_variables <- chosen_variables$rm.var
  
  complete_data <- complete_data[, c(covar, "condition", "time")]
  
  # Rename columns
  colnames(complete_data)[colnames(complete_data) == 'alder_tdato'] <- 'Age at donation'
  colnames(complete_data)[colnames(complete_data) == 'time'] <- 'Time to diagnosis'
  colnames(complete_data)[colnames(complete_data) == 'bd_grp'] <- "Blood donor group"
  colnames(complete_data)[colnames(complete_data) == 'physact_status'] <- "Physical activity"
  colnames(complete_data)[colnames(complete_data) == 'smoking_status'] <- "Smoking status"
  
  return(complete_data)
}

# Prepare data for summarizing
complete_data_full <- get_data(1)
complete_data_full$grp <- rep("Full data", nrow(complete_data_full))
complete_data_no_bdgrp1 <- get_data(2)
complete_data_no_bdgrp1$grp <- rep("No bd.grp 1", nrow(complete_data_no_bdgrp1))
complete_data_cm <- get_data(3)
complete_data_cm$grp <- rep("Complete metadata", nrow(complete_data_cm))

complete_data <- rbind(complete_data_full, complete_data_no_bdgrp1, complete_data_cm)
complete_data$grp <- as.factor(complete_data$grp)

# Create population summary table
pop <- tbl_strata(complete_data, strata = grp, ~.x %>%
             tbl_summary(by=condition) %>% modify_header(label = "**Covariate**") %>% bold_labels() %>%
               modify_caption("**Study population summary**")
)
pop

gt::gtsave(as_gt(pop), file = file.path(tempdir(), "population_summary.png"))

tbl <- tbl_summary(complete_data, by=data) %>% modify_header(label = "**Covariate**") %>% bold_labels() %>%
  modify_caption("**Full CRC group summary**")
tbl


# Prepare CM datagroup for cancer metadata summary tables
data <- read.csv("path3")

data$tdato_diag_time_years <- replace(data$tdato_diag_time_years, data$tdato_diag_time_years == 50.0, 10.0)


names(data)[names(data) == "tdato_diag_time_years"] <- "time"
data <- within(data, condition["tdato_diag_time_years" > 10.0 & "tdato_diag_time_years" < 10.99] <- "CRC")
data <- within(data, time[condition == "C"] <- 10.0)

complete_data <- select(data, 3:372)

complete_data$condition <- data$condition
complete_data$time <- data$time

complete_data <- within(complete_data, condition[condition == "CRC"] <- "Cases")
complete_data <- within(complete_data, condition[condition == "C"] <- "Controls")


covar <- c("sex", "alder_tdato", "smoking_status", "bmi", "physact_status", "bd_grp")

complete_data[covar] <- data[covar]

complete_data <- complete_data[, c(covar, "condition", "time")]

complete_data[c("metastasis", "location", "cancertype")] <- data[c("metastasis", "location", "cancertype")]

complete_data_1 <- data.frame(complete_data$metastasis)
complete_data_2 <- data.frame(complete_data$location)
complete_data_3 <- data.frame(complete_data$cancertype)

colnames(complete_data_1)[colnames(complete_data_1) == 'metastasis'] <- 'Cancer Staging'
colnames(complete_data_2)[colnames(complete_data_2) == 'cancertype'] <- 'Cancer Type'
colnames(complete_data_3)[colnames(complete_data_3) == 'location'] <- 'Cancer Location'

# Create cancer metadata summary tables
tbl <- tbl_summary(complete_data_1) %>% modify_header(label = "**Covariate**") %>% bold_labels() %>%
  modify_caption("**Population cancer staging summary**")
tbl2 <- tbl_summary(complete_data_2) %>% modify_header(label = "**Covariate**") %>% bold_labels() %>%
  modify_caption("**Population cancer location summary**")
tbl3 <- tbl_summary(complete_data_3) %>% modify_header(label = "**Covariate**") %>% bold_labels() %>%
  modify_caption("**Population cancer type summary**")
tbl
tbl2
tbl3

