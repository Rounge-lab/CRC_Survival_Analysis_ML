#! /usr/bin/Rscript

library(data.table)
library(dplyr)
library(reshape2)

test_id <- read.csv("test_id.csv")
tr_id <- read.csv("training_id.csv")

# full crc data
df_meta <- read.csv("path")
setnames(df_meta, c("X"), c("variable"))
df_meta <- data.table(df_meta)

df_count <- read.csv("path")
df_count <- dcast(melt(df_count, id.vars = "X"), variable ~ X)
df_count <- data.table(df_count)
df_crc <- df_count[df_meta, on = .(variable)]
setnames(df_crc, c("variable"), c("sampleID"))

df_crc_test <- df_crc[df_crc$JanusID %in% test_id$JanusID]
df_crc_tr <- df_crc[df_crc$JanusID %in% tr_id$JanusID]

write.csv(df_crc_test, file = "name")
write.csv(df_crc_tr, file = "name")

# No bdgrp1
df_meta <- read.csv("path")
setnames(df_meta, c("X"), c("variable"))
df_meta <- data.table(df_meta)

df_count <- read.csv("path")
df_count <- dcast(melt(df_count, id.vars = "X"), variable ~ X)
df_count <- data.table(df_count)
df_crc_bdgrp1 <- df_count[df_meta, on = .(variable)]
setnames(df_crc_bdgrp1, c("variable"), c("sampleID"))

df_crc_test_no_bdgrp1 <- df_crc_test[(df_crc_test$sampleID %in% df_crc_bdgrp1$sampleID),]
df_crc_tr_no_bdgrp1 <- df_crc_tr[(df_crc_tr$sampleID %in% df_crc_bdgrp1$sampleID),]

write.csv(df_crc_test_no_bdgrp1, file = "name")
write.csv(df_crc_tr_no_bdgrp1, file = "name")

# No bdgrp1 no NA
df_meta <- read.csv("path")
setnames(df_meta, c("X"), c("variable"))
df_meta <- data.table(df_meta)

df_count <- read.csv("path")
df_count <- dcast(melt(df_count, id.vars = "X"), variable ~ X)
df_count <- data.table(df_count)
df_crc_bdgrp1_NA <- df_count[df_meta, on = .(variable)]
setnames(df_crc_bdgrp1_NA, c("variable"), c("sampleID"))

df_crc_test_no_bdgrp1_no_NA <- df_crc_test[(df_crc_test$sampleID %in% df_crc_bdgrp1_NA$sampleID),]
df_crc_tr_no_bdgrp1_no_NA <- df_crc_tr[(df_crc_tr$sampleID %in% df_crc_bdgrp1_NA$sampleID),]

write.csv(df_crc_test_no_bdgrp1_no_NA, file = "name")
write.csv(df_crc_tr_no_bdgrp1_no_NA, file = "name")