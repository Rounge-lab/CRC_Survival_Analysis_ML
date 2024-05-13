library(data.table)
library(randomForestSRC)
library(dplyr)
library(reshape2)
library(ggplot2)
library(gridExtra)
library(grid)
library(egg)

# start with chosen variables:
output_distant <- read.csv("output_no_bdg1_no_NA_reg_time_Distant")
distant_miRNA <- output_distant$rm.var

#plot(output_distant$rsf.err, type="line")
#plot(1 -output_distant$rsf.err.auc, type="line")

idx <- which.min(output_distant$rsf.err)
output_distant <- output_distant[idx:nrow(output_distant), ]
output_distant <- output_distant$rm.var

output_regional <- read.csv("output_no_bdg1_no_NA_reg_time_Regional")
regional_miRNA <- output_regional$rm.var

#plot(output_regional$rsf.err, type="line")
#plot(1 -output_regional$rsf.err.auc, type="line")

idx <- which.min(output_regional$rsf.err)
output_regional <- output_regional[idx:nrow(output_regional), ]
output_regional <- output_regional$rm.var

output_localized <- read.csv("output_no_bdg1_no_NA_reg_time_localized")
localized_miRNA <- output_localized$rm.var

#plot(output_localized$rsf.err, type="line")
#plot(1 -output_localized$rsf.err.auc, type="line")

idx <- which.min(1-output_localized$rsf.err.auc)
output_localized <- output_localized[idx:nrow(output_localized), ]
output_localized <- output_localized$rm.var

output_distal <- read.csv("output_no_bdg1_no_NA_reg_time_distal")
distal_miRNA <- output_distal$rm.var

#plot(output_distal$rsf.err, type="line")
#plot(1 -output_distal$rsf.err.auc, type="line")

idx <- which.min(output_distal$rsf.err)
output_distal <- output_distal[idx:nrow(output_distal), ]
output_distal <- output_distal$rm.var

output_prox <- read.csv("output_no_bdg1_no_NA_reg_time_proximal")
prox_miRNA <- output_prox$rm.var

#plot(output_prox$rsf.err, type="line")
#plot(1 -output_prox$rsf.err.auc, type="line")

idx <- which.min(output_prox$rsf.err)
output_prox <- output_prox[idx:nrow(output_prox), ]
output_prox <- output_prox$rm.var

# Find common selected miRNAs
common <- intersect(output_distant, intersect(output_regional, intersect(output_localized, intersect(output_prox, output_distal))))

intersect(output_distal, output_prox)

intersect(output_distant, output_localized)

intersect(output_distant, output_regional)

intersect(output_regional, output_localized)

m <- (length(output_distal) +length(output_distant)+
            length(output_localized)+ length(output_prox) + length(output_regional)) / 5
m

m_dg <- (length(output_all) + length(output_no_bdgrp1) + length(output_no_bdgrp1_no_NA)) / 3
m_dg
