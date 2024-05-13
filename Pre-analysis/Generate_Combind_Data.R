#! /usr/bin/Rscript

library(data.table)
library(dplyr)
library(reshape2)

# Generate datasets for each data group using both train and test data

# full crc data
df_meta <- read.csv("path")
setnames(df_meta, c("X"), c("variable"))
df_meta <- data.table(df_meta)

df_count <- read.csv("path")
df_count <- dcast(melt(df_count, id.vars = "X"), variable ~ X)
df_count <- data.table(df_count)
df_crc <- df_count[df_meta, on = .(variable)]
setnames(df_crc, c("variable"), c("sampleID"))

write.csv(df_crc, file = "name")


# No bdgrp1
df_meta <- read.csv("path")
setnames(df_meta, c("X"), c("variable"))
df_meta <- data.table(df_meta)

df_count <- read.csv("path")
df_count <- dcast(melt(df_count, id.vars = "X"), variable ~ X)
df_count <- data.table(df_count)
df_crc_bdgrp1 <- df_count[df_meta, on = .(variable)]
setnames(df_crc_bdgrp1, c("variable"), c("sampleID"))

write.csv(df_crc_bdgrp1, file = "name")


# No bdgrp1 no NA
df_meta <- read.csv("path")
setnames(df_meta, c("X"), c("variable"))
df_meta <- data.table(df_meta)

df_count <- read.csv("path")
df_count <- dcast(melt(df_count, id.vars = "X"), variable ~ X)
df_count <- data.table(df_count)
df_crc_bdgrp1_NA <- df_count[df_meta, on = .(variable)]
setnames(df_crc_bdgrp1_NA, c("variable"), c("sampleID"))

write.csv(df_crc_bdgrp1_NA, file = "name")