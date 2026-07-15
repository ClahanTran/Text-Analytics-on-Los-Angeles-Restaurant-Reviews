# =============================================================================
# Step 2: Text Preprocessing
# Text Analytics on Los Angeles Restaurant Reviews
# =============================================================================
# Load user configuration (DATA_PATH, RESULTS_DIR, SEED, model params)
if (!exists('DATA_PATH')) source(file.path(getwd(), 'config.R'))

# Requires: reviews_raw loaded from 01_setup_load.R

# Slang normalization dictionary
slang_map <- c(
  "\bomg\b"     = "oh my god",
  "\blol\b"     = "laughing",
  "\btbh\b"     = "to be honest",
  "\bimo\b"     = "in my opinion",
  "\bimho\b"    = "in my humble opinion",
  "\bbtw\b"     = "by the way",
  "\bfav\b"     = "favorite",
  "\bdef\b"     = "definitely",
  "\bobs\b"     = "obviously",
  "\bgr8\b"     = "great",
  "\b4ever\b"   = "forever",
  "\bthx\b"     = "thanks",
  "\btks\b"     = "thanks",
  "\bu\b"       = "you",
  "\bur\b"      = "your",
  "\bw/\b"      = "with",
  "\bw/o\b"     = "without",
  "\bpls\b"     = "please",
  "\bplz\b"     = "please",
  "\bngl\b"     = "not gonna lie",
  "\bsup\b"     = "what is up"
)

# --- Clean review text ----------------------------------------------------
reviews_clean <- reviews_raw |>
  filter(!is.na(Comment)) |>
  mutate(
    comment_id = row_number(),
    # Impute missing price
    Price = replace_na(Price, "Unknown"),
    # 1. Lowercase
    text_clean = str_to_lower(Comment),
    # 2. Remove HTML tags
    text_clean = replace_html(text_clean),
    # 3. Normalize slang
    text_clean = str_replace_all(text_clean, slang_map),
    # 4. Expand contractions (e.g. "don't" -> "do not")
    text_clean = replace_contraction(text_clean),
    # 5. Remove punctuation and digits
    text_clean = str_replace_all(text_clean, "[^a-z\s]", " "),
    # 6. Collapse whitespace
    text_clean = str_squish(text_clean),
    # Derived metadata
    review_length   = str_count(Comment, "\S+"),
    char_length     = nchar(Comment),
    # Broad price tier label
    price_tier = case_when(
      Price == "$"    ~ "Budget",
      Price == "$$"   ~ "Moderate",
      Price == "$$$"  ~ "Upscale",
      Price == "$$$$" ~ "Fine Dining",
      TRUE            ~ "Unknown"
    ),
    # Binary sentiment label based on star rating
    sentiment_label = if_else(StarRating >= 4.0, "positive", "negative")
  )

cat("Cleaned dataset:", nrow(reviews_clean), "rows\n")

# --- Tokenize, remove stop words, lemmatize --------------------------------
data("stop_words")

tokens_lemma <- reviews_clean |>
  select(comment_id, RestaurantName, StarRating, price_tier,
         sentiment_label, Price, Style, text_clean) |>
  unnest_tokens(word, text_clean) |>
  anti_join(stop_words, by = "word") |>
  filter(nchar(word) > 2, !str_detect(word, "^\d+$")) |>
  mutate(word = lemmatize_words(word))

cat("Total tokens after cleaning:", nrow(tokens_lemma), "\n")
cat("Unique lemmas:", n_distinct(tokens_lemma$word), "\n")

# Save cleaned data for downstream steps
dir.create("data", showWarnings = FALSE)
saveRDS(reviews_clean, "data/reviews_clean.rds")
saveRDS(tokens_lemma,  "data/tokens_lemma.rds")
