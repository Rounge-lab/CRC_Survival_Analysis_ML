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

eval_times <- c(0.5, 1, 1.5, 2, 3, 5, 7, 9)
plotting_auc <- data.frame()
tp_df <- matrix(nrow =6, ncol=length(eval_times))
tp_se_df <- matrix(nrow=6, ncol=length(eval_times))
fp_df <- matrix(nrow= 6, ncol=length(eval_times))
fp_se_df <- matrix(nrow= 6, ncol=length(eval_times))
ppv_df <- matrix(nrow=6, ncol=length(eval_times))
ppv_se_df <- matrix(nrow=6, ncol=length(eval_times))
npv_df <- matrix(nrow=6, ncol=length(eval_times))
npv_se_df <- matrix(nrow=6, ncol=length(eval_times))

for(i in 1:6){
  if(i == 1 | i ==2){
    data <- read.csv("path1")
    }else if(i == 3 | i == 4){
      data <- read.csv("path2")}else{
        data <- read.csv("path3")}
  
  T_marker <- data$T - data$t
  delta_marker <- data$delta == 1
  if (i %% 2 == 0){
  risk_score <- data$risk_q1m0
  }else{
    risk_score <- data$risk_q4m2
  }
  
  data_set <- NULL
  if(i == 1){
    data_set <- "Full data group KSCR(q=1, m=0)"
  }else if(i == 2){
    data_set <- "Full data group KSCR(q=4, m=2)"}else if(i == 3){
      data_set <- "No_bdgrp1 data group KSCR(q=1, m=0)"
    }else if(i == 4){
      data_set <- "No bdgrp1 data group KSCR(q=4, m=2)"
    }else if(i == 5){
      data_set <- "Complete metadata group KSCR(q=1, m=0)"
    }else{
      data_set <- "Complete metadata group KSCR(q=4, m=2)"
    }
  
  tv_auc <- timeROC(T = T_marker, delta = delta_marker, marker = risk_score , times=eval_times, cause = 1, iid=TRUE)
  print(get.cindex(T_marker, delta_marker, risk_score))
  risk_quantiles <- quantile(risk_score, probs = seq(0, 1, by =.1))
  tp_list <- matrix(0, length(risk_quantiles), length(eval_times))
  fp_list <- matrix(0, length(risk_quantiles), length(eval_times))
  ppv_list <- matrix(0, length(risk_quantiles), length(eval_times))
  npv_list <- matrix(0, length(risk_quantiles), length(eval_times))
  for( j in 1:length(risk_quantiles)){ 
    sen <- SeSpPPVNPV(cutpoint=risk_quantiles[j], T = T_marker, delta = delta_marker, marker = risk_score, cause = 1, times = eval_times, iid=TRUE)
    tp_list[j,] <- sen$TP 
    fp_list[j,] <- sen$FP
    ppv_list[j,] <- sen$PPV
    npv_list[j, ] <- sen$NPV
  }
  tp_list <- data.frame(tp_list)
  fp_list <- data.frame(fp_list)

  J_stat <- tp_list - fp_list
  cp <- floor(mean(apply(J_stat, 2, function(x) c(max = max(x), ind = which.max(x)))[2,]))
  sen <- SeSpPPVNPV(cutpoint=risk_quantiles[cp], T = T_marker, delta = delta_marker, marker = risk_score, cause = 1, times = eval_times, iid=TRUE)
  
  tp_df[i, ] <- unname(sen$TP)
  tp_se_df[i, ] <- unname(sen$inference$vect_se_Se)
  fp_df[i, ] <- unname(sen$FP)
  fp_se_df[i, ] <- unname(sen$inference$vect_se_Sp1)
  ppv_df[i, ] <- unname(sen$PPV)
  ppv_se_df[i, ] <- unname(sen$inference$vect_se_PPV)
  npv_df[i, ] <- unname(sen$NPV)
  npv_se_df[i, ] <- unname(sen$inference$vect_se_NPV2)
  
  confidence_bands <- data.frame(unname(confint(tv_auc, level = 0.95, n.sim=3000)$CB_AUC))*0.01
  auc <- data.frame(unname(tv_auc$AUC))
  auc_to_plot <- cbind(confidence_bands, tv_auc$AUC)
  rownames(auc_to_plot) <- NULL
  auc_to_plot$data <- rep(data_set, nrow(auc_to_plot))
  auc_to_plot$t <- eval_times
  colnames(auc_to_plot) <- c("p05", "p95", "AUC", "data", "time")
  plotting_auc <- rbind(plotting_auc, auc_to_plot)
}

ggplot(plotting_auc, aes(x = time, y = AUC, groups=data)) +
  geom_line(col='blue') +
  geom_ribbon(aes(x = time, ymin = p05, ymax = p95), fill = "blue", alpha = 0.2) +
  geom_hline(yintercept = 0.5, size=0.2, color="black", linetype=2) +
  labs(x = "Time to diagnosis", y = "AUC") +
  ggtitle("Time varying AUC for KSCR models on holdout test set\n with 95% confidence bands by data group and covariate subset") +
  coord_cartesian(ylim = c(0.0, 1.0)) + scale_y_continuous(n.breaks=10) + 
  facet_wrap(~data, nrow=3, ncol=3) + theme(axis.text.x = element_text(size=10), 
                                            axis.text.y = element_text(size=10),
                                            strip.text = element_text(size=10))

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
tp <- paste0(round(as.matrix(tp_df), 2), " (", 
             +                   round(as.matrix(tp_se_df), 2), ")" )
tp <- matrix(tp, nrow=6, ncol=8)
tp <- data.frame(tp)


fp <- fp_df
fp<- paste0(round(as.matrix(fp_df), 2), " (", 
            +                   round(as.matrix(fp_se_df), 2), ")" )
fp <- matrix(fp, nrow=6, ncol=8)
fp <- data.frame(fp)

ppv <- ppv_df
ppv <- paste0(round(as.matrix(ppv_df), 2), " (", 
              +                   round(as.matrix(ppv_se_df), 2), ")" )
ppv <- matrix(ppv, nrow=6, ncol=8)
ppv <- data.frame(ppv)

npv <- npv_df
npv <- paste0(round(as.matrix(npv_df), 2), " (", 
              +                   round(as.matrix(npv_se_df), 2), ")" )
npv <- matrix(npv, nrow=6, ncol=8)
npv <- data.frame(npv)

row_name_list <- c("Full_q1m0", "Full_q4m2", "No_bdgrp1_q1m0", "No_bdgrp1_q4m2", "Complete_metadata_q1m0", "Complete_metadata_q4m2")
rownames(tp) <- row_name_list
rownames(fp) <- row_name_list
rownames(ppv) <-  row_name_list
rownames(npv) <- row_name_list


tp_str <- "name1"
fp_str <- "name2"
ppv_str <- "name3"
npv_str <- "name4"

write.csv(tp, tp_str)
write.csv(fp, fp_str)
write.csv(ppv, ppv_str)
write.csv(npv, npv_str)