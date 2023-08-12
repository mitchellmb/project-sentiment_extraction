# project-sentiment_extraction
Use natural language processing, boosted decision trees, and feature extraction to identify key business review sentiments.

This project looks at aggregated reviews from the open source Yelp data set from https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset.

The project starts with an initial data exploration notebook 0_Yelp_reviews_data_exploration.ipynb to figure out what the contents of the data are and how to use feature engineering to extract meaningful information, in particular, from the textual data.


Then, the project workflow follows:

1_json_to_csv.py - convert all JSON files to csv for easier reading/writing

2_review_text_fine_tune.py - fine tune bert-base-uncased on a small subset of textual data

3_batch_process.py - batch process feature engineering of chunks of data, following the original (TITLE) notebook exploration

   - import 3_1_text_preprocess.py to extract textual information
 
4_XGBoost_model.ipynb - fit a boosted tree model to classify results and examine most important model features

5_postmodel_analysis.ipynb - after modeling, extract the most important words & sentiments to drive business insights & value
