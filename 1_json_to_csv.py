'''
Convert reviews & business information from JSON to csv format
Follows the 0_Yelp_reviews_data_exploration.ipynb notebook exploration format
'''

#%%
import pandas as pd

#Load all 6.9m data points
review_json = pd.read_json(r'../archive/yelp_academic_dataset_review.json',
                     lines = True,
                     orient = 'columns',
                     chunksize = 100000)
review = pd.concat(review_json)
    
business_json = pd.read_json(r'../archive/yelp_academic_dataset_business.json',
                     lines = True,
                     orient = 'columns',
                     chunksize = 100000)
business = pd.concat(business_json)

#Rename and combine dataframes
review.rename(columns={'stars':'review stars'}, inplace=True)
business.rename(columns={'stars':'business stars'}, inplace=True)
business_reviews = review.merge(business, on='business_id')

#Drop unneeded features
drop_features = ['review_id', 'user_id', 'business_id', 'name', 'address',
                'city', 'is_open', 'attributes', 'hours', 'postal_code']

business_reviews.drop(columns=drop_features, inplace=True)
business_reviews = business_reviews.dropna().reset_index(drop=True) 

#Add in datetime features as separate columns, remove date column
business_reviews['year'] = [datetime.year for datetime in business_reviews['date']]
business_reviews['month'] = [datetime.month for datetime in business_reviews['date']]
business_reviews['hour'] = [datetime.hour for datetime in business_reviews['date']]
business_reviews.drop(columns='date', inplace=True)

#Save the data points as a new combined csv
business_reviews.to_csv(r'../csv_processed/0_collated_yelp_reviews.csv')