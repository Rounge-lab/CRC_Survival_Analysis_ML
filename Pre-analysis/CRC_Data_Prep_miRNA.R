
# Original work by Maja Sigerseth Jacobsen

##################################
######## LOADING PACKAGES ########
##################################

library(tidyverse)
library(lubridate)
library(ggsci)
library(viridis)
library(naniar)
library(limma)
library(PoiClaClu)
library(here)

########################################
######## LOADING JANUS RNA DATA ########
########################################

#filepaths
fp_caco_file <- paste("path")
fp_count_files <- paste("path")
fp_fixed_count_files <- paste("path")

### caco files ###

caco_full <- read_tsv(paste(fp_caco_file, "path", sep = ""))


#fixed count tables with ADF names

miRNA_count_ADF <- read_tsv(paste(fp_fixed_count_files, "name", sep = ""))


### stat files ###

stats <- read_tsv("path")

dup_ADF_sample <- read_tsv("path")




#############################################
########### CLEANUP ON META DATA ############
#############################################

caco_ADF <- caco_full %>% filter(str_detect(Study_ID, "ADF"))

#df to recode histology

crc_histology_df <- data.frame(histology = c(180,181,182,
                                             183,184,185,
                                             186,
                                             187,
                                             189,
                                             199,
                                             209),
                               cancertype = c("Ascending","Ascending","Ascending",
                                              "Transverse","Transverse","Transverse",
                                              "Descending",
                                              "Sigmoid",
                                              "Colon",
                                              "Rectosigmoid",
                                              "Rectum"), 
                               stringsAsFactors = F)

#df with metastasis data 

metastase_df <- data.frame(METASTASE1 = c("0","8","A","D","1","5","6","B","2","3","4","C","7","9"),
                           metastase = c(rep("Localized",2),rep("Regional",5),rep("Distant",4),rep("Unknown",3)),
                           stage_new_codes = c(rep("Early",2),rep("Locally Advanced",5),rep("Advanced",4),rep("Unknown",3)),
                           stringsAsFactors = F)

#metadata wrangling

caco_ADF_crc <- caco_full %>%
  filter(grepl("C18", ICD101)|grepl("C19", ICD101)|grepl("C20", ICD101)|site == "C") %>%
  filter(str_detect(Study_ID, "ADF")) %>% #filtering for ADF
  mutate(condition = as.factor(site)) %>% # site as factor
  mutate(physact_status = as.factor(ifelse(is.na(physact), NA,
                                    ifelse(physact %in% c(1, 2), "moderate",
                                          ifelse(physact %in% c(3, 4), "regularly", NA))))) %>% #making two categories for physical activity (1-inactive, 2-low, 3-medium, 4-high) - into "moderate" and "regularly" active
  mutate(smoking = as.factor(smoking)) %>% # smoking info
  mutate(smoking_status = as.factor(ifelse(is.na(smoking), NA,
                                           ifelse(smoking == 3, "never", 
                                                  ifelse(smoking %in% c(1, 2), "smoker", NA))))) %>% #categorical smoking (never is one category and current + former is another)
  mutate(bd_grp = as.factor(bd_grp)) %>% # BDg as factor
  mutate(tdato_diag_time_years = tdato_diag_time/52.177457) %>% 
  mutate(tdato_diag_time_date = as.POSIXct(TDATO) + as.numeric(tdato_diag_time)*7*24*60*60) %>%
  mutate(tdato_diag_time_months = as.integer(tdato_diag_time_years*12)) %>% 
  mutate(tdato_diag_time_years = ifelse(is.na(tdato_diag_time_years),50,tdato_diag_time_years)) %>% #put time to diagnosis to 50 years for controls  
  mutate(tdato_diag_time = as.factor(ifelse(is.na(tdato_diag_time),"C",tdato_diag_time))) %>% #time to diagnosis as factor
  mutate(alder_tdato_tertile = ntile(alder_tdato, 3)) %>% #age in tertile groups (intervals)
  mutate(sex = str_replace(SX, "K", "F")) %>% #recoding sex
  mutate(sex = as.factor(sex)) %>%
  mutate(BMI = as.factor(ifelse(bmi < 18.50,"Under",
                                ifelse(bmi < 25,"Normal",
                                       ifelse(bmi >= 30,"Obese","Overweight"))))) %>% #adding categorical BMI
  mutate(BMI_group = as.factor(ifelse(BMI == "Under" | BMI == "Normal", "normal", 
                                      ifelse(BMI == "Obese" | BMI == "Overweight", "over", NA)))) %>% #combining normal+under and obese+overweight to two categories
  dplyr::rename("sample" = "Study_ID") %>%
  mutate(histology = TOPOGRAFI_ICDO31) %>%
  left_join(crc_histology_df, by = "histology") %>% #add histology data
  left_join(metastase_df, by= "METASTASE1") %>% #add metastasis data 
  mutate(condition = replace_na(condition, "CRC")) %>% #add condition on the missing ones
  mutate(metastasis = ifelse(condition == "C","Control",metastase)) %>% 
  mutate(cancertype = ifelse(tdato_diag_time_years > 11 | condition == "C", "Control", cancertype)) %>%
  mutate(location = factor(case_when(
    cancertype %in% c("Ascending", "Transverse") ~ "proximal",
    cancertype %in% c("Descending", "Rectum", "Sigmoid", "Rectosigmoid") ~ "distal",
    cancertype == "Colon" ~ "unspecified",
    cancertype == "Control" ~ "control"
  ))) %>%   #splitting the cancer types to proximal and distal location
  select(sample, JanusID, TDATO, tdato_diag_time_date, bd_grp, histology, cancertype, location, ICD101, smoking, smoking_status, physact, physact_status, bmi, BMI_group, sex, metastasis, condition, serial_number, serial_y_n, tdato_diag_cat, tdato_diag_time, tdato_diag_time_years, tdato_diag_time_months, alder_tdato, alder_tdato_tertile) %>%
  arrange(nchar(sample), sample)

# Calculate the age interval for each tertile
tertile_intervals <- tapply(caco_ADF_crc$alder_tdato, caco_ADF_crc$alder_tdato_tertile, function(x) {
  min_age <- round(min(x))
  max_age <- round(max(x))
  paste(min_age, "-", max_age)
})

# Print the age intervals for each tertile
tertile_intervals

#df for metadata only for crc and with some serial variables
serial_samples <- caco_ADF_crc %>%
  rowwise() %>%
  mutate(JanusID = str_glue("ID_", str_pad(JanusID, "0",  width = 4, side = "left"), side = "left")) %>%
  mutate(JanusID = as_factor(JanusID)) %>%
  group_by(JanusID) %>% 
  mutate("Serial_total" = n()) %>% 
  ungroup() %>% 
  mutate("Serial" = Serial_total > 1) %>%
  filter(condition != "C") %>%
  arrange(desc(by=tdato_diag_time_years))


################################################
######## CLEANUP ON COUNT DATA #################
################################################

#Making a count dataset with only samples from caco (ADF, crc, c) to remove the rest.

names_caco_ADF_crc <- caco_ADF_crc[, 1]

miRNA_count_ADF_crc <- miRNA_count_ADF %>%
  pivot_longer(-ID, names_to = "sample") %>%
  inner_join(names_caco_ADF_crc, by = "sample") %>%
  pivot_wider(names_from = sample, values_from = value)

#Finding the samples that are in caco but not in count and remove them from caco. 
#It is 34 samples, that either had too low volume etc and were never sequenced.

miRNA_count_ADF_crc_long <- miRNA_count_ADF_crc %>%
  pivot_longer(-ID, names_to = "sample")

missing_count <- names_caco_ADF_crc %>% anti_join(miRNA_count_ADF_crc_long, by = "sample")

#creating the final metadata dataset without the missing samples

final_caco <- caco_ADF_crc[!(caco_ADF_crc$sample %in% missing_count$sample),] %>% 
  column_to_rownames(var = "sample")

#arranging the colnames (Sample_ID) in the count data in ascending order to be compatible with the metadata

final_mir_count <- miRNA_count_ADF_crc %>% 
  select(ID, num_range("ADF", range = 1:1751)) %>%
  column_to_rownames(var = "ID")

all(rownames(final_caco) %in% colnames(final_mir_count))
all(rownames(final_caco) == colnames(final_mir_count))

#filtering for miRNAs with counts higher than 10 in more than 20% of the samples

final_mir_count_filtered <- final_mir_count %>% 
  rownames_to_column("mir") %>%
  filter(apply(.[,-1], 1, function(x) sum(x>10) >= length(x)*0.20)) %>%
  column_to_rownames("mir")

### Save final datasets ###

#write.csv(final_caco, here("janus_crc_metadata.csv"), row.names = TRUE)
#saveRDS(final_caco, here("data", "janus_crc_metadata.rds"))
#write.csv(final_mir_count_filtered, here("janus_crc_countdata.csv"), row.names = TRUE)
#saveRDS(final_mir_count_filtered, here("data", "janus_crc_countdata.rds"))


######################################################################
### CLEANUP ON COUNTDATA AND METADATA - REMOVE BLOOD DONOR GROUP 1 ###
######################################################################

metadata_bdgrp1 <- final_caco %>%
  subset(bd_grp == "Grp1")

to_remove <- rownames(metadata_bdgrp1)

metadata_no_bdgrp1 <- final_caco %>%
  subset(bd_grp != "Grp1")

countdata_no_bdgrp1 <- final_mir_count_filtered[, !colnames(final_mir_count_filtered) %in% to_remove]

all(rownames(metadata_no_bdgrp1) == colnames(countdata_no_bdgrp1))

### Save final datasets ###

write.csv(metadata_no_bdgrp1, here("janus_crc_metadata_no_bdgrp1.csv"), row.names = TRUE)
#saveRDS(metadata_no_bdgrp1, here("data", "janus_crc_metadata_no_bdgrp1.rds"))
write.csv(countdata_no_bdgrp1, here("janus_crc_countdata_no_bdgrp1.csv"), row.names = TRUE)
#saveRDS(countdata_no_bdgrp1, here("data", "janus_crc_countdata_no_bdgrp1.rds"))

##########################################################################################
### CLEANUP ON COUNTDATA AND METADATA - REMOVE SAMPLES MISSING DATA ON BMI AND SMOKING ###
##########################################################################################

metadata_no_bdgrp1_no_NA <- metadata_no_bdgrp1 %>%
  filter(!is.na(smoking_status)) %>%
  filter(!is.na(BMI_group))

metadata_BMI <- metadata_no_bdgrp1 %>%
  filter(is.na(BMI_group)) %>%
  rownames()

metadata_smoking <- metadata_no_bdgrp1 %>%
  filter(is.na(smoking_status)) %>%
  rownames()

countdata_no_bdgrp1_no_NA <- countdata_no_bdgrp1[, !colnames(countdata_no_bdgrp1) %in% c(metadata_BMI, metadata_smoking), ]

all(rownames(metadata_no_bdgrp1_no_NA) == colnames(countdata_no_bdgrp1_no_NA))

### Save final datasets ###

#write.csv(metadata_no_bdgrp1_no_NA, here("janus_crc_metadata_no_bdgrp1_no_NA.csv"), row.names = TRUE)
#saveRDS(metadata_no_bdgrp1_no_NA, here("data", "janus_crc_metadata_no_bdgrp1_no_NA.rds"))
#write.csv(countdata_no_bdgrp1_no_NA, here("janus_crc_countdata_no_bdgrp1_no_NA.csv"), row.names = TRUE)
#saveRDS(countdata_no_bdgrp1_no_NA, here("data", "janus_crc_countdata_no_bdgrp1_no_NA.rds"))


metadata_no_NA <- final_caco %>%
  filter(!is.na(smoking_status)) %>%
  filter(!is.na(BMI_group))

metadata_BMI_2 <- final_caco %>%
  filter(is.na(BMI_group)) %>%
  rownames()

metadata_smoking_2 <- final_caco %>%
  filter(is.na(smoking_status)) %>%
  rownames()

countdata_no_NA <- final_mir_count_filtered[, !colnames(final_mir_count_filtered) %in% c(metadata_BMI_2, metadata_smoking_2), ]

all(rownames(metadata_no_NA) == colnames(countdata_no_NA))

### Save final datasets ###

#write.csv(metadata_no_NA, here("janus_crc_metadata_no_NA.csv"), row.names = TRUE)
#saveRDS(metadata_no_NA, here("data", "janus_crc_metadata_no_NA.rds"))
#write.csv(countdata_no_NA, here("janus_crc_countdata_no_NA.csv"), row.names = TRUE)
#saveRDS(countdata_no_NA, here("data", "janus_crc_countdata_no_NA.rds"))

############################
########## PLOTS ###########
############################


#cancertypes

reorder_size <- function(x) {
  factor(x, levels = names(sort(table(x), decreasing = TRUE)))
}

ggplot(final_caco, aes(x = reorder_size(cancertype), fill = cancertype)) +
  geom_bar(stat = "count") +
  theme_bw() +
  scale_fill_npg() +
  scale_x_discrete(expand = c(0,0)) +
  theme(axis.text.x = element_blank()) +
  labs(title = "Number of individuals within each cancer type divided by sex", x = "cancertype", y = "count") + 
  facet_grid(~ sex) 

ggplot(serial_samples, aes(x = reorder_size(cancertype), fill = cancertype)) + #used serial_samples df because of no controls here
  geom_bar(stat = "count") +
  theme_bw() +
  theme(axis.text.x = element_blank()) +
  scale_fill_npg() +
  scale_x_discrete(expand = c(0,0)) +
  labs(title = "Number of individuals within each cancer type divided by metastase", x = "cancertype", y = "count") + 
  facet_grid(~ metastasis) 


#blood donor group

ggplot(final_caco, aes(x = bd_grp, fill = bd_grp)) +
  geom_bar() +
  scale_fill_npg() +
  theme_bw() +
  geom_text(stat = "count", aes(label = ..count..), position = position_stack(vjust = 0.5))
  labs(title = "Number of individuals from each blood donor group", x = "Blood donor group", y = "count")

ggsave(here("plots", "barplot_bdgrp_numbers.png")) 

  
ggplot(final_caco, aes(x = reorder_size(cancertype), fill = cancertype)) + 
  geom_bar(stat = "count") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_fill_npg() +
  scale_x_discrete(expand = c(0,0)) +
  labs(title = "Number of individuals within each cancer type from each blood donor group", x = "cancertype", y = "count") + 
  facet_grid(~ bd_grp) 

ggplot(final_caco, aes(x = condition, fill = condition)) + 
  geom_bar(stat = "count") +
  geom_text(stat = "count", aes(label = ..count..), position = position_stack(vjust = 0.5)) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_fill_npg() +
  scale_x_discrete(expand = c(0,0)) +
  labs(title = "Number of cases and controls from each sex", x = "condition", y = "count") +
  facet_grid(~ sex)


ggsave(here("plots", "barplot_caco_sex.png"))

#missing data within blood donor groups

missing_df <- final_caco %>%
  select(smoking, physact, BMI_group, bd_grp)

gg_miss_var(missing_df, facet = bd_grp) + labs(y = "number of NA")

final_caco %>%
  group_by(bd_grp) %>%
  miss_var_summary() %>%
  filter(variable == "smoking")

final_caco %>%
  group_by(bd_grp) %>%
  miss_var_summary() %>%
  filter(variable == "physact")

final_caco %>%
  group_by(bd_grp) %>%
  miss_var_summary() %>%
  filter(variable == "BMI")

# Plot CRC time to diagnosis for serial and not serial samples for each cancer location

p_dist <- serial_samples %>% 
  mutate(JanusID = factor(JanusID, levels = (serial_samples %>% 
                                               group_by(JanusID) %>% 
                                               summarise(maxtime = max(tdato_diag_time_years)) %>% 
                                               arrange(desc(maxtime)) %>% 
                                               pull(JanusID)))) %>%
  filter(location == "distal") %>% 
  ggplot(aes(x=JanusID, y=tdato_diag_time_years, label=tdato_diag_time_years, colour=Serial)) +
  geom_segment(aes(y = 0,
                   x = JanusID,
                   yend = tdato_diag_time_years,
                   xend = JanusID), color = 'grey') +
  geom_point(stat='identity', size=2)  +
  labs(title = "distal crc", 
       y = "Time to diagnosis (Years)",
       colour = "serial samples") +
  scale_y_reverse(breaks = c(0,1,2,3,4,5,6,7,8,9,10)) +
  scale_x_discrete(name = "Individual", labels = NULL) +
  scale_color_manual(values  = c("#D55E00", "#56B4E9"), labels = c("No", "Yes")) +
  coord_flip() 
p_dist + theme_minimal() + 
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 12),
        axis.title = element_text(face="bold", size = 14),
        axis.text = element_text(size = 12),
        panel.grid.minor = element_blank())

p_prox <- serial_samples %>% 
  mutate(JanusID = factor(JanusID, levels = (serial_samples %>% 
                                               group_by(JanusID) %>% 
                                               summarise(maxtime = max(tdato_diag_time_years)) %>% 
                                               arrange(desc(maxtime)) %>% 
                                               pull(JanusID)))) %>%
  filter(location == "proximal") %>% 
  ggplot(aes(x=JanusID, y=tdato_diag_time_years, label=tdato_diag_time_years, colour=Serial)) +
  geom_segment(aes(y = 0,
                   x = JanusID,
                   yend = tdato_diag_time_years,
                   xend = JanusID), color = 'grey') +
  geom_point(stat='identity', size=2)  +
  labs(title = "Proximal crc",
       y = "Time to diagnosis (Years)",
       colour = "serial samples") +
  scale_y_reverse(breaks = c(0,1,2,3,4,5,6,7,8,9,10)) +
  scale_x_discrete(name = "Individual", labels = NULL) +
  scale_color_manual(values  = c("#D55E00", "#56B4E9"), labels = c("No", "Yes")) +
  coord_flip() 
p_prox + theme_minimal() + theme(legend.position = "bottom", 
                            axis.title = element_text(face="bold"),
                            panel.grid.minor = element_blank())


p_unsp <- serial_samples %>% 
  mutate(JanusID = factor(JanusID, levels = (serial_samples %>% 
                                               group_by(JanusID) %>% 
                                               summarise(maxtime = max(tdato_diag_time_years)) %>% 
                                               arrange(desc(maxtime)) %>% 
                                               pull(JanusID)))) %>%
  filter(location == "unspecified") %>% 
  ggplot(aes(x=JanusID, y=tdato_diag_time_years, label=tdato_diag_time_years, colour=Serial)) +
  geom_point(stat='identity', size=2)  +
  labs(title = "crc unspecified location",
       y = "Time to diagnosis (Years)",
       colour = "serial samples") +
  scale_y_reverse(breaks = c(0,1,2,3,4,5,6,7,8,9,10)) +
  scale_x_discrete(name = "Individual", labels = NULL) +
  scale_color_manual(values  = c("#D55E00", "#56B4E9"), labels = c("No", "Yes")) +
  coord_flip() 
p_unsp + theme_minimal() + theme(legend.position = "bottom", 
                                 axis.title = element_text(face="bold"),
                                 panel.grid.minor = element_blank())

#time to diagnosis, serial samples and the blood donor groups

bd_p <- serial_samples  %>% 
  mutate(JanusID = factor(JanusID, levels = (serial_samples %>% 
                                               group_by(JanusID) %>% 
                                               summarise(maxtime = max(tdato_diag_time_years)) %>% 
                                               arrange(desc(maxtime)) %>% 
                                               pull(JanusID)))) %>%
  ggplot(aes(x = JanusID, y = tdato_diag_time_years, label = tdato_diag_time_years, colour = bd_grp, shape = Serial)) +
  geom_point(stat='identity', size=4)  +
  labs(y = "Time to diagnosis (Years)",
       colour = "Janus blood donor groups:") +
  scale_y_reverse(breaks = c(0,1,2,3,4,5,6,7,8,9,10)) +
  scale_x_discrete(name = "Individual", labels = NULL)  +
  coord_flip() 
bd_p + theme_minimal() + theme(legend.position = "bottom", 
                                axis.title = element_text(face="bold"),
                                panel.grid.minor = element_blank()) 



