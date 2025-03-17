#code referenced and adapted from the following sources
## Adyatama & Christian (2020) (https://rpubs.com/Argaadya/topic_lda)
## Jones (2021) (https://cran.r-project.org/web/packages/textmineR/vignettes/c_topic_modeling.html)
## McDonnell (2016) (https://rpubs.com/RobertMylesMc/Bayesian-IRT-ideal-points-with-Stan-in-R)
## Markwick (2018) (https://dm13450.github.io/2018/01/18/DIC.html)

#load necessary packages for text cleaning and analysis



library(tidyverse)

library(tm)
library(corpus)
library(tidytext)
library(textclean)
library(lubridate)
library(hunspell)
library(stopwords)
library(textmineR)
library(scales)
library(jsonlite)
library(stopwords)
library(stargazer)
library(rstan)
library(randomcoloR)


library(ggwordcloud)

setwd("/Users/eclin/Desktop/ML_Project")

#load oral argument data
oa = read_csv("/Users/eclin/Desktop/ML_Project/alloralargs.csv")

#oa_df = as.data.frame(oa)

#clean oral argument data 
#(note, some preliminary cleaning is already performed in python during text extraction)

#check what problems may be present in text
sample = (oa$all_text[1])
sample = unlist(strsplit(sample, "(?<=[[:punct:]])\\s(?=[A-Z])", perl=T))

check_text(sample)

clean_oa = oa %>% mutate(clean_text = all_text %>%
                           replace_non_ascii() %>% 
                           tolower() %>%
                           str_remove_all("petitioner|respondent|case|court|justice|question|companyofficial") %>% 
                           replace_emoticon() %>%
                           add_missing_endmark() %>%
                           replace_symbol() %>% 
                           replace_contraction() %>%
                           replace_word_elongation() %>%
                           str_replace_all("[[:punct:]]", " ") %>%
                           make_plural() %>%
                           str_squish() %>%
                           str_trim(), 
                         
                         clean_justice_text = justice_text %>%
                           replace_non_ascii() %>% 
                           tolower() %>%
                           str_remove_all("petitioner|respondent|case|court|justice|question|companyofficial") %>% 
                           replace_emoticon() %>%
                           add_missing_endmark() %>%
                           replace_symbol() %>% 
                           replace_contraction() %>%
                           replace_word_elongation() %>%
                           str_replace_all("[[:punct:]]", " ") %>%
                           make_plural() %>%
                           str_squish() %>%
                           str_trim(),
                         
                         clean_filer_text = filer_text %>%
                           replace_non_ascii() %>% 
                           tolower() %>%
                           str_remove_all("petitioner|respondent|case|court|justice|question|companyofficial") %>% 
                           replace_emoticon() %>%
                           add_missing_endmark() %>%
                           replace_symbol() %>% 
                           replace_contraction() %>%
                           replace_word_elongation() %>%
                           str_replace_all("[[:punct:]]", " ") %>%
                           make_plural() %>%
                           str_squish() %>%
                           str_trim()
                         
                         )


#create some general summary statistics 
all_text_length = sapply(strsplit(clean_oa$clean_text, " "), length)
justice_length = sapply(strsplit(clean_oa$clean_justice_text, " "), length)
filer_length = sapply(strsplit(clean_oa$clean_filer_text, " "), length)

all_text_sum = all_text_length %>% summary()
all_text_sum
          
justice_sum = justice_length %>% summary()
justice_sum
filer_sum = filer_length %>% summary()
filer_sum

#tokenize text for further processing + remove stop words
create_stems <- function(term) {
  stems = hunspell_stem(term)[[1]]
  
  if (length(stems) == 0){
    stem = term
  }
  else{
    stem = stems[[length(stems)]]
  }
  return (stem)
}

#set stopwords list
#we choose a relatively comprehensive stopwords list since our text data is relatively messy
stopwords_list = get_stopwords(language = "en", source = "stopwords-iso")

all_text_tokens = clean_oa %>% 
  unnest_tokens(output = "word", input = clean_text) %>%
  anti_join(stopwords_list) %>%
  mutate(word = text_tokens(word, stemmer = create_stems) %>% as.character()) %>%
  drop_na(word) %>%
  count(caseId, word)

dtm_alltext = all_text_tokens %>% cast_dtm(document = caseId, term = word, value = n)
inspect(dtm_alltext)
dtm_alltext

#filter out terms that appear in over 90 percent of documents to reduce dimensionality
frequency = findFreqTerms(dtm_alltext, lowfreq = 5, highfreq = nrow(dtm_alltext)*0.95)
dtm_freqfiltered = dtm_alltext[, frequency]
dtm_freqfiltered
inspect(dtm_freqfiltered)

lda_input_matrix = Matrix(as.matrix(dtm_freqfiltered), sparse = TRUE)



# fit LDA model -----------------------------------------------------------

#set range of topics to check
check_range = 30

#define parameters where defaults are set following lauderdale & clark (2014)
beta = 0.01

#set iterations
#ideally, we set this number to 2000, and allow burnin to be 1000
#however, we cut this number in half to reduce computational time
iters = 1000 
burnin_num = 500

set.seed(46)

#create vectors to store models
# store_lda_models = vector(mode = "list", check_range - 1)

#set model path
model_path = "/Users/eclin/Desktop/ML_Project/LDAModels/"

#check fit of number of topics from 2 to 30
# for (i in 2:check_range){
#   topic_num = i
#   alpha = topic_num / 50
#   store_idx = i-1
#   
#   print(paste("working on num_topics:", topic_num))
#   start.time <- Sys.time()

# 
#   lda_model = FitLdaModel(lda_input_matrix, k = topic_num, iterations = iters,
#                           burnin = burnin_num,
#                           calc_likelihood = TRUE,
#                           calc_coherence = TRUE,
#                           calc_r2 = TRUE)
#   store_lda_models[[store_idx]] = lda_model
#   
#   #save model as file
#   file_name = paste0(model_path, "numtopics", topic_num, ".rds")
#   saveRDS(lda_model,file = file_name)
#   
#   
#   end.time <- Sys.time()
#   time.taken <- end.time - start.time
#   print(paste("Time taken:", round(time.taken, 3)))
# }



#evaluate fit of topic numbers

max_coherence_vec = rep(NA, length(store_lda_models))
min_coherence_vec = rep(NA, length(store_lda_models))
mean_coherence_vec = rep(NA, length(store_lda_models))

num_topics_vec = rep(NA, length(store_lda_models))
for (i in 1:length(store_lda_models)){
  model = store_lda_models[[i]]
  num_topics = i + 1
  
  coherence = model$coherence
  
  max_coherence = max(coherence)
  min_coherence = min(coherence)
  avg_coherence = mean(coherence)
  
  max_coherence_vec[i] = max_coherence
  min_coherence_vec[i] = min_coherence
  mean_coherence_vec[i] = avg_coherence
  num_topics_vec[i] = num_topics
}

coherence_df = cbind(num_topics_vec, max_coherence_vec, min_coherence_vec, avg_coherence)

#graph maximum coherence
coherence_df %>%
  ggplot( aes(x=num_topics_vec, y=max_coherence_vec)) +
  geom_line( color="grey") +
  geom_point(shape=21, color="lightblue3", fill="lightblue3", size=2) +
  ggtitle("Maximum Coherence by Number of Topics") +
  xlab("Number of Topics") + 
  ylab("Maximum Coherence")

#graph minimum coherence
coherence_df %>%
  ggplot( aes(x=num_topics_vec, y=min_coherence_vec)) +
  geom_line( color="grey") +
  geom_point(shape=21, color="lightpink2", fill="lightpink2", size=2) +
  ggtitle("Minimum Coherence by Number of Topics") +
  xlab("Number of Topics") + 
  ylab("Mininum Coherence")

#graph minimum coherence
coherence_df %>%
  ggplot( aes(x=num_topics_vec, y=mean_coherence_vec)) +
  geom_line( color="grey") +
  geom_point(shape=21, color="orange", fill="orange", size=2) +
  ggtitle("Average Coherence by Number of Topics") +
  xlab("Number of Topics") + 
  ylab("Average Coherence")


#NOTE: peak of min and max coherence at 25 topics
lda_24 = store_lda_models[[23]]
lda_25 = store_lda_models[[24]]

#get top terms in each topic area
seq_24 = seq(1, 24)
col_names24 = paste("Topic", seq_24)

seq_25 = seq(1, 25)
col_names25 = paste("Topic", seq_25)

top_terms24 = GetTopTerms(lda_24$phi, 3) %>% as.data.frame()
colnames(top_terms24) = col_names24


top_terms25 = GetTopTerms(lda_25$phi, 3) %>% as.data.frame()
colnames(top_terms25) = col_names25

print(xtable(top_terms24[,1:10], type = "latex"),include.rownames = FALSE )
print(xtable(top_terms24[,11:20], type = "latex"), include.rownames = FALSE)
print(xtable(top_terms24[,21:24], type = "latex"), include.rownames = FALSE)

print(xtable(top_terms25[,1:10], type = "latex"),include.rownames = FALSE )
print(xtable(top_terms25[,11:20], type = "latex"), include.rownames = FALSE)
print(xtable(top_terms25[,21:25], type = "latex"), include.rownames = FALSE)

#get documents that belong to each topic
#set LDA topics to use
lda_to_use = lda_25
lda_idx_to_use = 24
top_terms_tu = top_terms25

topic_terms = rep(NA, ncol(top_terms_tu))
for (i in 1:ncol(top_terms_tu)){
  relcol = top_terms_tu[i]
  allterms = paste(unlist(relcol), collapse = ", ")
  topic_terms[i] = allterms
}

rel_cols_topics = paste0("t_", seq_25)
case_topics = lda_to_use$theta %>% as.data.frame() %>% rownames_to_column("caseId")
case_dates = oa %>% select(year, caseId, issueArea)
case_topics_dates = merge(case_topics, case_dates, on = "caseId")
case_td_names = c("caseID", topic_terms, "year", "issueArea")

match_topic_to_name = cbind(seq_25, rel_cols_topics, topic_terms)


#compare against spaeth dataset
num_topic_spaeth = seq(1, 14)
name_topic_spaeth = c("Criminal Procedure", "Civil Rights", "First Amendment", "Due Process",
                      "Privacy", "Attorneys", "Unions", "Economic Activity", "Judicial Power",
                      "Federalism", "Interstate Relations", "Federal Taxation", "Misc.", "Private Action")

spaeth_topics = cbind(num_topic_spaeth, name_topic_spaeth)
print(xtable(spaeth_topics), include.rownames = FALSE)

spaeth_v_lda = merge(case_topics_dates, spaeth_topics,
                     by.x = "issueArea", 
                     by.y = "num_topic_spaeth",
                     all.x = TRUE)
max_svl = colnames(spaeth_v_lda[rel_cols_topics])[apply(spaeth_v_lda[rel_cols_topics], 1, which.max)]
spaeth_v_lda = cbind(spaeth_v_lda, max_svl)


long_df_spaethlda = spaeth_v_lda %>% select(caseId, issueArea, name_topic_spaeth, max_svl) %>% 
  pivot_longer(c(max_svl), names_to = "max_svl",
  values_to = "topic")

long_df_spaethlda = merge(long_df_spaethlda, match_topic_to_name, by.x = "topic", 
                          by.y = "rel_cols_topics",
                          all.x = TRUE)

comp_table = as.data.frame(xtabs(~name_topic_spaeth+topic_terms, data = long_df_spaethlda))
order = c("Misc.", "Federal Taxation", "Interstate Relations", "Federalism", "Judicial Power",
          "Economic Activity", "Unions", "Attorneys", "Privacy", "Due Process", "First Amendment", 
          "Civil Rights", "Criminal Procedure", "Private Action")

comp_table$name_topic_spaeth = factor(comp_table$name_topic_spaeth, levels = rev(order))

comp_plot = ggplot(comp_table, aes(x = topic_terms, y = name_topic_spaeth, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "lightgray", high = "darkblue") +  # Darker colors for higher counts
  labs(title = "Spaeth et al. v. LDA", x = "LDA Categories", y = "Spaeth Categories", fill = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

comp_plot

ggsave(comp_plot, file="/Users/eclin/Desktop/ML_Project/Visuals/CompPlot.png", width = 20, height = 20, units = "cm")

#compare evolution of topics over time

long_df = case_topics_dates %>% pivot_longer(c(rel_cols_topics), names_to = "topic",
                                             values_to = "theta")
long_df = merge(long_df, match_topic_to_name, by.x = "topic", 
                by.y = "rel_cols_topics")

overtime = long_df %>% group_by(year, topic_terms, topic) %>% summarise(theta = mean(theta)) 

file_path_overtime = "/Users/eclin/Desktop/ML_Project/Overtime/"
for (i in 1:length(rel_cols_topics)){
  t = rel_cols_topics[i]
  overtime_topic = overtime %>% filter(topic == t)
  
  plot = overtime_topic %>% ggplot(aes(year, theta, fill = topic_terms, color = topic_terms)) +
    geom_line() +
    geom_point(show.legend = F) +
    scale_color_manual(values = c(randomColor(count = 1, luminosity = "dark"))) +
    theme(legend.position = "top")
  
  filename = paste0(file_path_overtime, "topic_", t, ".png")
  ggsave(plot, file=filename, width = 14, height = 10, units = "cm")
  
}


# perform ideal point analysis --------------------------------------------
#get posterior mixture from lda model, store in relevant vectors
length_set = 29
topic_mix_vec = vector(mode = "list", length_set)
word_dist_vec = vector(mode = "list", length_set)
for (i in 1:length_set){
  lda_model = store_lda_models[[i]]
  
  doc_theta = lda_model$theta
  doc_phi = lda_model$phi
  
  topic_mix_vec[[i]] = doc_theta
  word_dist_vec[[i]] = doc_phi
  
}
lda_model$theta


#create function in stan that takes in topic mixtures and calculates justice ideal points
# Declare code for stan model
stan_model_code = "
data{
 int<lower = 1> J; //justices
 int<lower = 1> C; // cases
 int<lower = 1> N; // number of observed votes in a case

 int<lower = 1, upper = C> case_idx[N]; // case indicides for observed votes
 int<lower = 0, upper = J> justice_idx[N]; // justice indices
 int votes[N]; // votes
 int<lower = 1> num_topics; //number of topics

 matrix[C, num_topics] topics; // Topic proportions in each case

}

parameters{
  matrix[J, num_topics] ideal_points; //

  real alpha[C];
  real beta[C];

}

model{
for (j in 1:J){
  for (num in 1:num_topics){
      ideal_points[j, num] ~ normal(0,1);
  }
}

for (c in 1:C){
  alpha[c] ~ normal(0,4);
  beta[c] ~ normal(0, 4);
}

for (n in 1:N){
  int c = case_idx[n]; //
  int j = justice_idx[n]; //

  if (votes[n] != -1){
    real vote_prob = alpha[c] + beta[c] * dot_product(ideal_points[j], topics[c]);
    votes[n] ~ bernoulli_logit(vote_prob);//
  } 
}
}
"

#Compile stan model
sm = stan_model(model_code = stan_model_code)


####FUNCTION####
ideal_point_estm <- function(total_justices, vote_matrix, lda_model, 
                             stan_model,
                             iter, chains, warmup, cores, seed){
  distribution = lda_model$theta
  num_topics = ncol(lda_model$theta)
  
  post_mix = distribution %>% 
    as.data.frame() %>%
    set_names(paste("Topic", 1:num_topics)) %>%
    rownames_to_column("case")
  
  topics_only = as.matrix(post_mix %>% select(-case))
  
  stan_data <- list(
    J = total_justices,
    C = nrow(vote_matrix),
    N = length(votes_full),
    case_idx = case_idx_full,
    justice_idx = justice_idx_full,
    votes = votes_full,
    num_topics = num_topics,
    topics = topics_only
  )
  
  fit <- sampling(stan_model, data = stan_data, iter = iter, chains = chains, 
                  warmup = warmup, cores = cores, seed = seed)
  
  return(fit)
  
}

#load in and clean vote data
vote_df = read_csv("/Users/eclin/Desktop/ML_Project/justice_vote_matrix.csv")
vote_df = vote_df %>% select((order(colnames(vote_df))))

votes_only = vote_df %>% select(-c(caseId))
vote_matrix = as.matrix(votes_only)

#create some information about justices to store
justices = (paste(colnames(votes_only), collapse = ", "))
justice_list = unlist(as.list(strsplit(justices, ",")[[1]]))

justice_idx = seq(1: length(justice_list))

justice_match = cbind(justice_idx, justice_list)

#set params for IRT
total_justices = length(justice_list)

#Create the full justice_idx, including all 39 justices, with NA for non-voters
justice_idx_full <- rep(1:total_justices, nrow(vote_matrix))
case_idx_full <- rep(1:nrow(vote_matrix), each = total_justices)

# Replace missing values (NA) with -1 for votes (treat as "missing")
votes_full <- as.vector(vote_matrix)
votes_full[is.na(votes_full)] <- -1  # Treat NA as missing (-1 for missing)


#create a vector to store ideal point models
set_length = 29
#store_ideal_point_models2 = vector(mode = "list", set_length)

save_ipm_basepath = "/Users/eclin/Desktop/ML_Project/IdealPointModels/"

for (i in 1:set_length){
  lda_model = store_lda_models[[i]]
  num_topics = (i+1)
  
  print(paste("working on num_topics:", num_topics))
  start.time <- Sys.time()
  
  ideal_point_fit = ideal_point_estm(total_justices = total_justices, 
                                     vote_matrix = vote_matrix,
                                     lda_model = lda_model,
                                     stan_model = sm,
                                     iter = 2000,
                                     warmup = 500,
                                     chains = 4, 
                                     cores = 4,
                                     seed = 46)
  
  store_ideal_point_models[[i]] = ideal_point_fit
  
  #save model as file
  file_name = paste0(save_ipm_basepath, "IPM_numtopics", num_topics, ".rds")
  saveRDS(ideal_point_fit,file = file_name)
  
  end.time <- Sys.time()
  duration = end.time - start.time
  print(paste("Time taken:", round(duration, 3)))
  
}
}

#fitted model with 25 topics
fit = store_ideal_point_models2[[lda_idx_to_use]]

#extract ideal points from fit
ip = extract(fit)$ideal_points
meanip = apply(ip, c(2,3), mean)
quantile_low = apply(ip, c(2,3), quantile, 0.025)
quantile_up = apply(ip, c(2,3), quantile, 0.975)

ideal_topics = lda_idx_to_use + 1
ideal_point_file_path = "/Users/eclin/Desktop/ML_Project/IdealPointGraphs/"
for (i in 1:ideal_topics){
  ip_df <- data.frame(
    Justice = justice_match,
    Topic = i,
    Lower = as.vector(quantile_low[,i]),
    Mean = as.vector(meanip[,i]),
    Upper = as.vector(quantile_up[,i])
  )
  
  topic_name = as.character(match_topic_to_name[i,3])
  plot_title = paste("Supreme Court Justices' Ideal Points for Topic:", topic_name)
  
  file_name = paste0(ideal_point_file_path, "topic", i, ".png")
  
  plot = ggplot(ip_df, aes(x = Justice.justice_list, y = Mean, ymin = Lower, ymax = Upper, color = as.factor(Topic))) +
    geom_pointrange(position = position_dodge(width = 0.5), size = 0.7) + 
    geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
    coord_flip() +
    labs(title = plot_title,
         x = "Justice", 
         y = "Ideal Point") +
    scale_color_manual(values = c(randomColor(count = 1, luminosity = "dark"))) +
    theme(axis.text.y = element_text(size = 10),
          axis.text.x = element_text(size = 10),
          legend.position = "none")
  
  plot
  
  
  ggsave(plot, file=file_name, width = 20, height = 20, units = "cm")
  
}





