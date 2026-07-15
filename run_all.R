# =============================================================================
# run_all.R
# Master script: runs all 10 steps of the LA Restaurant Reviews analysis
# in the correct order. All outputs are saved to RESULTS_DIR (see config.R).
# =============================================================================
# USAGE:
#   Rscript run_all.R          # from the repo root in a terminal
#   source('run_all.R')        # from an interactive R session
# =============================================================================

# --- 0. Load user configuration ------------------------------------------
source('config.R')
dir.create(RESULTS_DIR, showWarnings = FALSE)

cat('=======================================================\n')
cat(' Text Analytics on LA Restaurant Reviews\n')
cat(' Full pipeline: Steps 1-10\n')
cat('=======================================================\n\n')

run_step <- function(n, label, script) {
  cat(sprintf('[Step %02d] %s ...\n', n, label))
  t <- system.time(source(script, local = FALSE))
  cat(sprintf('         Done in %.1f sec\n\n', t['elapsed']))
}

# --- Step 1 & 2: Setup, data loading, preprocessing ---------------------
run_step(1, 'Setup & data loading',  'scripts/01_setup_load.R')
run_step(2, 'Text preprocessing',    'scripts/02_preprocessing.R')

# --- Step 3: Exploratory text analysis -----------------------------------
run_step(3, 'Exploratory text analysis', 'scripts/03_eda.R')

# --- Step 4: Sentiment analysis ------------------------------------------
run_step(4, 'Sentiment analysis (AFINN / Bing / NRC / sentimentr)',
         'scripts/04_sentiment_analysis.R')

# --- Step 5: Keyword extraction (TF-IDF) ---------------------------------
run_step(5, 'Keyword extraction (TF-IDF)', 'scripts/05_keyword_extraction.R')

# --- Step 6: Word embeddings (Word2Vec + GloVe) --------------------------
run_step(6, 'Word embeddings (Word2Vec + GloVe)', 'scripts/06_word_embeddings.R')

# --- Step 7: Clustering / segmentation -----------------------------------
# Note: doc_embeddings must exist from Step 6
run_step(7, 'Clustering & segmentation (k-means, k=5)', 'scripts/07_clustering.R')

# --- Step 8: Topic modeling (LDA) ----------------------------------------
run_step(8, 'Topic modeling (LDA, k=6)', 'scripts/08_topic_modeling.R')

# --- Step 9: ML modeling -------------------------------------------------
run_step(9, 'Statistical & ML modeling (LASSO + Random Forest)',
         'scripts/09_ml_modeling.R')

# --- Step 10: Business insights ------------------------------------------
run_step(10, 'Business insights & recommendations', 'scripts/10_business_insights.R')

cat('=======================================================\n')
cat(' All steps complete.\n')
cat(sprintf(' Results saved to: %s/\n', RESULTS_DIR))
cat('=======================================================\n')
