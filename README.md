# Text Analytics on Los Angeles Restaurant Reviews

A full text analytics pipeline applied to 2,381 Yelp reviews across 240 top-ranked
Los Angeles restaurants. The project covers ten analytical steps from raw text
preprocessing through machine-learning modeling and actionable business recommendations.

**Team:** Clahan Tran, Abdullah Khan, Kristin Henry, Dane Alban  
**Language:** R 4.5+  
**Date:** July 2026

---

## Repository structure

```
.
├── config.R                  # <-- Edit this first: set DATA_PATH for your machine
├── run_all.R                 # Master script: runs all 10 steps in order
├── scripts/
|   ├── 01_setup_load.R       # Step 1  : Package installation & data loading
|   ├── 02_preprocessing.R    # Step 2  : Text cleaning, tokenisation, lemmatisation
|   ├── 03_eda.R              # Step 3  : Exploratory text analysis & word clouds
|   ├── 04_sentiment_analysis.R # Step 4: AFINN / Bing / NRC / sentimentr
|   ├── 05_keyword_extraction.R # Step 5: TF-IDF by restaurant, price tier, rating
|   ├── 06_word_embeddings.R  # Step 6  : Word2Vec + GloVe embeddings, UMAP
|   ├── 07_clustering.R       # Step 7  : K-means clustering (k=5) on embeddings
|   ├── 08_topic_modeling.R   # Step 8  : LDA topic modeling (k=6)
|   ├── 09_ml_modeling.R      # Step 9  : LASSO + Random Forest prediction
|   └── 10_business_insights.R # Step 10: Findings & recommendations
├── notebooks/
|   ├── full_analysis_report.qmd   # Quarto source: all 10 steps
|   ├── full_analysis_report.html  # Rendered HTML report
|   ├── progress_report.qmd        # Quarto source: Steps 1-3 progress report
|   └── progress_report.html       # Rendered HTML progress report
├── results/                  # Auto-generated: plots and output files
└── archive (1)/              # Raw data (not tracked by git -- see Data section)
```

---

## Quickstart: reproducing the analysis

### 1. Clone the repository

```bash
git clone https://github.com/ClahanTran/Text-Analytics-on-Los-Angeles-Restaurant-Reviews.git
cd Text-Analytics-on-Los-Angeles-Restaurant-Reviews
```

### 2. Obtain the data

Download the dataset and place the CSV file anywhere on your machine.
The file is named:

```
top 240 restaurants recommanded in los angeles 2.csv
```

### 3. Configure the data path

Open `config.R` and update `DATA_PATH` to point to the CSV on your machine:

```r
# config.R
DATA_PATH <- "C:/path/to/your/data/top 240 restaurants recommanded in los angeles 2.csv"
```

All other settings (seed, model parameters, output directory) can also be
adjusted in `config.R`.

### 4. Run the full pipeline

**Option A — one command (terminal):**
```bash
Rscript run_all.R
```

**Option B — interactive R session:**
```r
source('run_all.R')
```

**Option C — run individual steps:**
```r
source('config.R')          # always run this first
source('scripts/01_setup_load.R')
source('scripts/02_preprocessing.R')
# ... etc.
```

Scripts must be run **in order** (01 → 10) because each step depends on
objects created by prior steps.

### 5. View the HTML report

Open `notebooks/full_analysis_report.html` in any browser, or view it online:

https://htmlpreview.github.io/?https://github.com/ClahanTran/Text-Analytics-on-Los-Angeles-Restaurant-Reviews/blob/main/notebooks/full_analysis_report.html

---

## R package dependencies

All packages are installed automatically by `scripts/01_setup_load.R`.
The table below lists them for reference.

| Package | Purpose |
|---------|---------|
| `tidyverse` | Data wrangling and visualization |
| `tidytext` | Text tokenisation and TF-IDF |
| `textstem` | Lemmatisation |
| `textclean` | HTML removal, contraction expansion |
| `textdata` | Sentiment lexicons (AFINN, Bing, NRC) |
| `sentimentr` | Negation-aware sentence sentiment |
| `wordcloud` | Word cloud visualizations |
| `topicmodels` | LDA topic modeling |
| `tm` | Document-term matrix construction |
| `word2vec` | Word2Vec embeddings |
| `text2vec` | GloVe embeddings |
| `umap` | Dimensionality reduction for visualization |
| `cluster` | Silhouette scores for cluster evaluation |
| `glmnet` | LASSO regression and classification |
| `ranger` | Random forest modeling |
| `patchwork` | Combining ggplot2 figures |
| `ggrepel` | Non-overlapping plot labels |
| `scales` | Formatting axes and labels |

**R version:** 4.5.1 or later recommended.

---

## Pipeline overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_setup_load.R` | Install packages, load data, check missingness |
| 2 | `02_preprocessing.R` | Lowercase, HTML removal, slang normalization, stop-word removal, lemmatization |
| 3 | `03_eda.R` | Word frequencies, review length distributions, word clouds |
| 4 | `04_sentiment_analysis.R` | AFINN, Bing, NRC, sentimentr; aggregate by restaurant and price tier |
| 5 | `05_keyword_extraction.R` | TF-IDF by restaurant, price tier, and rating band; log-odds ratio |
| 6 | `06_word_embeddings.R` | Word2Vec + GloVe (100 dims); document embeddings; UMAP visualizations |
| 7 | `07_clustering.R` | K-means (k=5) on document embeddings; cluster profiling |
| 8 | `08_topic_modeling.R` | LDA (k=6) with perplexity selection; topic-cluster heatmap |
| 9 | `09_ml_modeling.R` | LASSO + Random Forest: predict star rating and price tier |
| 10 | `10_business_insights.R` | Evidence-backed recommendations for restaurant managers |

---

## Key findings

- **86%** of reviews are AFINN-positive; AFINN agrees with star-rating labels **79%** of the time.
- **Service & Wait Times** reviews score 7x lower in sentiment than food-focused reviews (AFINN 1.4 vs 10.1).
- **Five review clusters** identified: Fine Dining & Upscale, Detailed Food Descriptions,
  Service Complaints, Noodle & Asian Cuisine, Short Casual Reviews.
- **Six LDA topics**: Overall Praise, Asian Cuisine & Flavors, Ambiance & Location,
  Service & Wait Times, Dining Experience, Food Detail (Western).
- Star rating prediction R² ≈ 0.12 (compressed rating range limits signal);
  price tier classification: **75.6%** accuracy vs. 73.4% baseline.

---

## Notes

- The `archive (1)/` data folder is listed in `.gitignore` and not tracked by git.
  You must supply the CSV yourself (see Step 2 above).
- The `results/` folder is also gitignored; plots are generated locally on each run.
- Rendered HTML reports in `notebooks/` **are** tracked so collaborators can view
  results without re-running the pipeline.
- The `tidymodels` package is **not** used due to a namespace conflict with `rlang`
  in the current session; `glmnet` and `ranger` are used directly instead.
