#!/usr/bin/env Rscript

library(dplyr)
library(randomForestSRC)
library(data.table)

set.seed(10)

args = commandArgs(trailingOnly=TRUE)
if (length(args) != 5) {
  stop("exactly 5 arguments must be supplied (input file).csv", call.=FALSE)
}

data <- read.csv(paste("data_sets/data_files", args[1], sep = "/"))
file_name <- args[2]
id_samples <- read.csv(paste("data_sets/data_files",args[3], sep = "/"))$id
run <- as.integer(args[5])
options(rf.cores = as.integer(args[4]))

data$T <- data$T - data$t
data_ts <- data[data$id %in% c(id_samples), ]
data_tr <- data[!(data$id %in% c(id_samples)), ]
data_ts <- data_ts[order(data_ts$id),]
data_ts <- subset(data_ts, select = -c(t, id))
data_tr <- subset(data_tr, select = -c(t, id))


data_tr$event <- as.integer(data_tr$event)
data_ts$event <- as.integer(data_ts$event)

## run survival trees and calculate number of times each variable splits a node
xvar.used <- rfsrc(Surv(T, event) ~ ., data=data_tr, ntree=1000, nodedepth = 6, perf.type="none",
                   var.used="all.trees", mtry = Inf, nsplit = 100, seed=run)$var.used

## calculate minimal depth with supervision
## use number of times variable splits to guide random feature selection
os <- rfsrc(Surv(T, event) ~ ., data=data_tr, ntree=1000, xvar.wt = xvar.used, 
            importance= FALSE, seed=run)
mst <- max.subtree(os, sub.order = TRUE)
mds <- mst$order[, 1]

# write chosen variables to file
sup_chosen_vars <- data.frame(mds[mds <= mst$threshold])
colnames(sup_chosen_vars) <- "minimal_dept"
file_name_cov <- paste("data_sets/covars", file_name, sep="/")
mm_names <- paste(file_name_cov, "rsf_chosen_var_supervised", sep="_")
write.csv(sup_chosen_vars, file = paste(mm_names, "csv", sep="."))

#chosen_vars <- names(mds[mds <= mst$threshold])
#chosen_vars <- append(chosen_vars, c("T", "event"))
#data_tr_chosen_var <- data_tr[, c(chosen_vars)]
#data_ts_chosen_var <- data_ts[, c(chosen_vars)]

# fit rsf mod using all variables
#tuning <- tune(Surv(T, event) ~ ., data_tr_chosen_var)
#rsf_mod <- rfsrc(Surv(T, event) ~., data=data_tr_chosen_var, ntree=1000 ,importance= FALSE, nodesize = tuning$optimal[1], mtry=tuning$optimal[2] ,seed=10)

#pred = predict.rfsrc(rsf_mod, data_ts_chosen_var, outcome = "test")
#file_name_pred <- paste("results/perliminary", file_name, sep="/")
#pred_names = paste(paste(file_name_pred, "rsf_pred", sep = "_"), "csv", sep = ".")
#write.csv(pred$predicted, file = pred_names)
