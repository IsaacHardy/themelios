import csv
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv('Articles_Temp_Table.csv')

to_drop = [
    'status',
    'editorial_author',
    'editorial_subtitle',
    'expiration_date',
    'editorial_series_newspring',
    'editorial_metatitle',
    'editorial_metadescription'
]

df.drop(to_drop, axis=1, inplace=True)

df['editorial_legacy_body'] = df['editorial_legacy_body'].str.replace(r'<[^>]+>', '')

df.fillna('', inplace=True)
df['title']

vector = TfidfVectorizer(stop_words='english')
vector_data = vector.fit_transform(df['editorial_legacy_body'])
vector_data.shape

cosine_sim = linear_kernel(vector_data, vector_data)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    content_indices = [i[0] for i in sim_scores]
    return df.iloc[content_indices].to_json(orient='records')