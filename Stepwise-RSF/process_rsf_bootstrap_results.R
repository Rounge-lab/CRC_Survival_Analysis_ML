library(data.table)
library(randomForestSRC)
library(dplyr)
library(reshape2)
library(ggplot2)
library(gridExtra)
library(grid)
library(egg)

# start with selected variables across data groups:
output_all <- read.csv("path1")

all_miRNA <- output_all$rm.var

# get index of minimum and remove all miRNAs before it
idx <- which.min(output_all$rsf.err)
output_all <- output_all[idx:nrow(output_all), ]
output_all <- output_all$rm.var

output_no_bdgrp1 <- read.csv("path2")
idx <- which.min(output_no_bdgrp1$rsf.err)
output_no_bdgrp1 <- output_no_bdgrp1[idx:nrow(output_no_bdgrp1), ]
output_no_bdgrp1 <- output_no_bdgrp1$rm.var

output_no_bdgrp1_no_NA <- read.csv("path3")
idx <- which.min(output_no_bdgrp1_no_NA$rsf.err)
output_no_bdgrp1_no_NA <- output_no_bdgrp1_no_NA[idx:nrow(output_no_bdgrp1_no_NA), ]
output_no_bdgrp1_no_NA <- output_no_bdgrp1_no_NA$rm.var


# Create a matrix to represent the table
table_data <- matrix(0, nrow = length(all_miRNA), ncol = 4)
colnames(table_data) <- c("all_miRNA", "all", "no_bdgrp1", "no_bdgrp1_no_NA")
table_data[, 1] <- all_miRNA
table_data[, 2] <- as.numeric(all_miRNA %in% output_all)
table_data[, 3] <- as.numeric(all_miRNA %in% output_no_bdgrp1)
table_data[, 4] <- as.numeric(all_miRNA %in% output_no_bdgrp1_no_NA)

table_df <- as.data.frame(table_data)

# Create a function to highlight specific cells
highlight_cells <- function(x, subtitle) {
  g <- ggplot(x, aes(x = "", y = all_miRNA)) +
    geom_tile(aes(fill = as.factor(x[,2])), color = "white") +
    scale_fill_manual(values = c("gray", "red")) +
    theme_void() +
    theme(legend.position = "none",
          panel.grid = element_blank(),
          axis.text = element_text(size = 8)) +
    labs(title = subtitle)
  return(g)
}

# Create separate plots for each list
plot_list1 <- highlight_cells(table_df[, c(1, 2)], "all data")
plot_list2 <- highlight_cells(table_df[, c(1, 3)], "no bdgrp1")
plot_list3 <- highlight_cells(table_df[, c(1, 4)], "no bdgrp1 no NA's")

# Arrange plots in a grid
grid.arrange(plot_list1, plot_list2, plot_list3, ncol = 3, 
             top=textGrob("Chosen miRNA's", gp = gpar(fontsize = 16, fontface = "bold")),
             layout_matrix = cbind(c(1), c(2), c(3)))

# Calculate the number of common elements between each pair of sets
common_1_2 <- length(intersect(output_all, output_no_bdgrp1))
common_1_3 <- length(intersect(output_all, output_no_bdgrp1_no_NA))
common_2_3 <- length(intersect(output_no_bdgrp1, output_no_bdgrp1_no_NA))
c1_2 <- intersect(output_all, output_no_bdgrp1)
c1_3 <- intersect(output_all, output_no_bdgrp1_no_NA)
c2_3 <- intersect(output_no_bdgrp1, output_no_bdgrp1_no_NA)
c1_2_3 <- intersect(intersect(c1_2, c1_3), c2_3)
common_1_2_3 <- length(c1_2_3)

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
print(c1_2_3) 

common_mirnas <- c(intersect(intersect(output_no_bdgrp1, output_all), output_no_bdgrp1_no_NA))



# evaluate bootstrap results

# C-index
all.c <- read.csv("path4")
colnames(all.c)[colnames(all.c) == "x"] <- "all_data_rna_covars"
all.c$all_data_rna <- read.csv("path5")$x
all.c <- subset(all.c, select = -X)
all.c$all_data_covars <- read.csv("path6")$x
all.c$nobdgrp1_rna_covars <- read.csv("path7")$x
all.c$nobdgrp1_rna <- read.csv("path8")$x
all.c$nobdgrp1_covars <- read.csv("path9")$x
all.c$nobdgrp1_no_NA_rna_covars <- read.csv("path10")$x
all.c$nobdgrp1_no_NA_rna <- read.csv("path11")$x
all.c$nobdgrp1_no_NA_covars <- read.csv("path12")$x


c_lower_025 <- apply(all.c, 2, quantile, probs=0.025)
c_upper_975 <- apply(all.c, 2, quantile, probs=0.975)
c_mean <- apply(all.c, 2, mean)

c_df <- data.frame(t(rbind(c_mean,c_lower_025, c_upper_975)))
write.csv(c_df, file="path13")

meltData <- melt(all.c)
p <- ggplot(meltData, aes(factor(variable), value)) 
p + geom_boxplot() + facet_wrap(~variable, scale="free") + labs(title = "C_R-index prediction error")

# Time-varyig AUC
all.auc_rc <- read.csv("path14")
all.auc_rc <- subset(all.auc_rc, select = -X)
names(all.auc_rc) <- all.auc_rc[1,]
all.auc_rc <- all.auc_rc[-1,]
all.auc_r <- read.csv("path15")
all.auc_r <- subset(all.auc_r, select = -X)
names(all.auc_r) <- all.auc_r[1,]
all.auc_r <- all.auc_r[-1,]
all.auc_c <- read.csv("path16")
all.auc_c <- subset(all.auc_c, select = -X)
names(all.auc_c) <- all.auc_c[1,]
all.auc_c <- all.auc_c[-1,]

bdgrp1.auc_rc <- read.csv("path17")
bdgrp1.auc_rc <- subset(bdgrp1.auc_rc, select = -X)
names(bdgrp1.auc_rc) <- bdgrp1.auc_rc[1,]
bdgrp1.auc_rc <- bdgrp1.auc_rc[-1,]
bdgrp1.auc_r <- read.csv("path18")
bdgrp1.auc_r <- subset(bdgrp1.auc_r, select = -X)
names(bdgrp1.auc_r) <- bdgrp1.auc_r[1,]
bdgrp1.auc_r <- bdgrp1.auc_r[-1,]
bdgrp1.auc_c <- read.csv("path19")
bdgrp1.auc_c <- subset(bdgrp1.auc_c, select = -X)
names(bdgrp1.auc_c) <- bdgrp1.auc_c[1,]
bdgrp1.auc_c <- bdgrp1.auc_c[-1,]

bdgrp1_NA.auc_rc <- read.csv("path20")
bdgrp1_NA.auc_rc <- subset(bdgrp1_NA.auc_rc, select = -X)
names(bdgrp1_NA.auc_rc) <- bdgrp1_NA.auc_rc[1,]
bdgrp1_NA.auc_rc <- bdgrp1_NA.auc_rc[-1,]
bdgrp1_NA.auc_r <- read.csv("path21")
bdgrp1_NA.auc_r <- subset(bdgrp1_NA.auc_r, select = -X)
names(bdgrp1_NA.auc_r) <- bdgrp1_NA.auc_r[1,]
bdgrp1_NA.auc_r <- bdgrp1_NA.auc_r[-1,]
bdgrp1_NA.auc_c <- read.csv("path22")
bdgrp1_NA.auc_c <- subset(bdgrp1_NA.auc_c, select = -X)
names(bdgrp1_NA.auc_c) <- bdgrp1_NA.auc_c[1,]
bdgrp1_NA.auc_c <- bdgrp1_NA.auc_c[-1,]

create_auc_plot <- function(df, title){
  
  mean<- colMeans(df)
  p05 <-apply(df, 2, function(x) quantile(x, 0.5))
  p95 <-apply(df, 2, function(x) quantile(x, 0.95))
  df_plot <- data.frame(
    time.points = factor(names(df), levels= names(df)),
    mean = mean,
    p05 = p05,
    p95 = p95,
    title <- rep(title, length(time.points))
    )

  return(df_plot)
}

plot.all.rc <- create_auc_plot(all.auc_rc, "Full: miRNAs +\nconfounders")
plot.all.r <- create_auc_plot(all.auc_r, "Full: miRNAs")
plot.bdgrp1.rc <- create_auc_plot(bdgrp1.auc_rc, "No bdgrp1: miRNAs +\nconfounders")
plot.bdgrp1.r <- create_auc_plot(bdgrp1.auc_r, "No bdgrp1: miRNAs")
plot.bdgrp1.NA.rc <- create_auc_plot(bdgrp1_NA.auc_rc, "Complete metadata: miRNAs +\nconfounders")
plot.bdgrp1.NA.r <- create_auc_plot(bdgrp1_NA.auc_r, "Complete metadata:\nmiRNAs")
plot.bdgrp1.NA.c <- create_auc_plot(bdgrp1_NA.auc_c, "Complete metadata:\nconfounders")

y.grob <- textGrob("AUC", gp=gpar(col="black", fontsize=12), rot=90)
x.grob <- textGrob("Time to diagnosis", gp=gpar(col='black', fontsize=12))
title.grob <- textGrob("Time varying AUC with bootstrapped confidence intervals", gp=gpar(col='black', fontsize=15))

ggarrange(plot.all.rc, plot.all.r, plot.bdgrp1.rc, plot.bdgrp1.r, plot.bdgrp1.NA.rc, plot.bdgrp1.NA.r, plot.bdgrp1.NA.c, 
          left=y.grob, bottom=x.grob, top=title.grob)

plotting_auc <- plot.all.rc <- create_auc_plot(all.auc_rc, "Full: miRNAs +\nconfounders")
plotting_auc <- rbind(plotting_auc, create_auc_plot(all.auc_r, "Full: miRNAs"))
plotting_auc <- rbind(plotting_auc, create_auc_plot(bdgrp1.auc_rc, "No bdgrp1: miRNAs +\nconfounders"))
plotting_auc <- rbind(plotting_auc, create_auc_plot(bdgrp1.auc_r, "No bdgrp1: miRNAs"))
plotting_auc <- rbind(plotting_auc, create_auc_plot(bdgrp1_NA.auc_rc, "Complete metadata: miRNAs +\nconfounders"))
plotting_auc <- rbind(plotting_auc, create_auc_plot(bdgrp1_NA.auc_r, "Complete metadata:\nmiRNAs"))
plotting_auc <- rbind(plotting_auc, create_auc_plot(bdgrp1_NA.auc_c, "Complete metadata:\nconfounders"))

p <- ggplot(df_plot, aes(x = time.points, y = mean, group=1)) +
  geom_line(color = "blue") +
  geom_ribbon(aes(ymin = p05, ymax = p95), fill = "gray", alpha = 0.5) +
  labs(x = "Time to diagnosis", y = "AUC") 
  ggtitle("Time varying AUC with bootstrapped confidence intervals") +
  coord_cartesian(ylim = c(0.4, 0.7)) + facet_wrap(~title, nrow=3, ncol=3)
  
  

# Using ALl data
# compare selected miRNAs
output_all <- read.csv("path23")

all_miRNA <- output_all$rm.var

idx <- which.min(output_all$rsf.err)
output_all <- output_all[idx:nrow(output_all), ]
output_all <- output_all$rm.var

output_no_bdgrp1 <- read.csv("path24")
idx <- which.min(output_no_bdgrp1$rsf.err)
output_no_bdgrp1 <- output_no_bdgrp1[idx:nrow(output_no_bdgrp1), ]
output_no_bdgrp1 <- output_no_bdgrp1$rm.var

output_no_bdgrp1_no_NA <- read.csv("path25")
idx <- which.min(output_no_bdgrp1_no_NA$rsf.err)
output_no_bdgrp1_no_NA <- output_no_bdgrp1_no_NA[idx:nrow(output_no_bdgrp1_no_NA), ]
output_no_bdgrp1_no_NA <- output_no_bdgrp1_no_NA$rm.var

c1_2 <- intersect(output_all, output_no_bdgrp1)
c1_3 <- intersect(output_all, output_no_bdgrp1_no_NA)
c2_3 <- intersect(output_no_bdgrp1, output_no_bdgrp1_no_NA)
c1_2_3 <- intersect(intersect(c1_2, c1_3), c2_3)
common_1_2_3 <- length(c1_2_3)

c1_2_3


