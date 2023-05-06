# import libraries
import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from underthesea import word_tokenize, pos_tag, sent_tokenize
import warnings
from gensim import corpora, models, similarities
import re


# load data
df1 = pd.read_csv('Products_1.csv')
df2 = pd.read_csv('Products_2.csv')

# combine 2 dataframes
df = pd.concat([df1, df2], ignore_index=True)

# df = pd.read_csv('Products.csv')
# df['rating'] = df['rating'].astype(float).round(1)

STOP_WORD_FILE = 'vietnamese-stopwords.txt'
with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
    stop_words = file.read()

stop_words = stop_words.split('\n')

# function to process text
def process_text_(text):
    # convert to lower case
    text = text.lower()
    # remove '\n' character
    text = text.replace('\n', ' ')
    # remove numbers '0-9'
    text = re.sub(r'[0-9]+', '', text)
    # remove punctuation
    text = text.replace('[^\w\s]', ' ')
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for char in punctuation:
        text = text.replace(char, ' ')
    # remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # remove leading and trailing spaces
    text = text.strip()
    # remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


##LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()


# load model gensim
tfidf_gensim = models.TfidfModel.load('tfidf.model')
index_gensim = similarities.SparseMatrixSimilarity.load('index.model')
dictionary_gensim = corpora.Dictionary.load('dictionary.model')


# define function to get top 10 similar products
def sim_products(text):
    # process text
    text = process_text_(text)
    # tokenize text
    text = word_tokenize(text, format="text")
    # convert text to vector
    text = dictionary_gensim.doc2bow(text.split())
    # get similarity score
    sim = index_gensim[tfidf_gensim[text]]
    # get top 10 similar products
    top_10_similar_products = sorted(enumerate(sim), key = lambda x: x[1], reverse = True)[1:11]
    # get product_id of top 10 similar products
    top_10_similar_products_id = [df.iloc[i[0]].product_id for i in top_10_similar_products]
    return top_10_similar_products_id

# define function to get top 10 similar products from tfidf_gensim, product_id
def sim_products_tfidf_gensim(product_id):
    # get index of product_id
    products_wt = df[df['product_id'] == product_id].products_wt.values
    # convert to string
    products_wt = str(products_wt)
    # process text
    # products_wt = process_text_(products_wt)
    # tokenize text
    # products_wt = word_tokenize(products_wt, format="text")
    # convert text to vector
    products_wt = dictionary_gensim.doc2bow(products_wt.split())
    # get similarity score
    sim = index_gensim[tfidf_gensim[products_wt]]
    # get top 10 similar products
    top_10_similar_products = sorted(enumerate(sim), key = lambda x: x[1], reverse = True)[1:11]
    # get product_id of top 10 similar products
    top_10_similar_products_id = [df.iloc[i[0]].product_id for i in top_10_similar_products]
    return top_10_similar_products_id


# read data from user_recs.parquet
# df_ = pd.read_parquet('user_recs.parquet')
# # load df_user_user_id.parquet use pandas
# df_user_user_id = pd.read_parquet('df_user_user_id.parquet')

# load df_product_id_product_id_idx.parquet use pandas
df_product_id_product_id_idx = pd.read_parquet('df_product_id_product_id_idx.parquet')



# def get_recommendations_list_idx(user_id_idx):
#     list_product_id_idx = []
#     for i in range(5):
#         idx = df_[df_['user_id_idx']==user_id_idx]['recommendations'].index.values[0]
#         list_product_id_idx.append(df_[df_['user_id_idx']==user_id_idx]['recommendations'][idx][i]['product_id_idx'])
#     return list_product_id_idx


# # define function to get list of product_id from user
# def get_recommendations_list(user):
#     # get user_id_idx from user
#     user_id_idx = int(df_user_user_id[df_user_user_id['user']==user]['user_id_idx'].values[0])
#     # get list of product_id_idx from user_id_idx
#     list_product_id_idx = get_recommendations_list_idx(user_id_idx)
#     # get list of product_id from list_product_id_idx
#     list_product_id = []
#     for i in list_product_id_idx:
#         list_product_id.append(df_product_id_product_id_idx[df_product_id_product_id_idx['product_id_idx']==i]['product_id'].values[0])
#     return list_product_id

# load new_user_recs.parquet
df_new_user_recs = pd.read_parquet('new_user_recs.parquet')


# define function to get list of product_id from user
def get_recommendations_list(user):
    # get product_id_idx from df_new_user_recs
    list_product_id_idx = []
    for i in range(5):
        list_product_id_idx.append(df_new_user_recs[df_new_user_recs['user']==user]['recommendations'].values[0][i]['product_id_idx'])
    # get list of product_id from list_product_id_idx
    list_product_id = []
    for i in list_product_id_idx:
        list_product_id.append(df_product_id_product_id_idx[df_product_id_product_id_idx['product_id_idx']==i]['product_id'].values[0])
    return list_product_id

# define function to recommend products from product_id_new
def show_recommendations(product_id_new):
    # write title product_id_new
    st.write('Product_id:', product_id_new)
    
    # get image and title of top 10 similar products_id_new
    lst_link_image = df[df['product_id'].isin(sim_products_tfidf_gensim(product_id_new))]['image'].tolist()
    
    # get list product_id of lst_link = product_id
    lst_link_product = df[df['product_id'].isin(sim_products_tfidf_gensim(product_id_new))]['link'].tolist()
    
    # get list title of lst_link = product_name
    lst_title = df[df['product_id'].isin(sim_products_tfidf_gensim(product_id_new))]['product_name'].tolist()
    
    # get list product_id of lst_link = product_id
    lst_product_id = df[df['product_id'].isin(sim_products_tfidf_gensim(product_id_new))]['product_id'].tolist()
    
    # concat lst_title and lst_product_id
    lst_title = [lst_title[i] + '  /  Product ID: ' + str(lst_product_id[i]) for i in range(len(lst_title))]
    
    # show image and title of top 10 similar products_id_new
    
    # display image, title, price, rating of product
    for i in range(len(lst_link_image)):
        if str(lst_link_image[i]) != 'nan':
            # show image                                    
            st.markdown(f'<a href="{lst_link_product[i]}" target="_blank"><img src="{lst_link_image[i]}" width="500" height="500"></a>',
                        # get product_id
                        unsafe_allow_html=True
                        )

            st.button('Recommendations for this product', key = lst_product_id[i], on_click = lambda id=lst_product_id[i]: show_recommendations(product_id_new=id))
                                    
            # display title of product
            st.write(lst_title[i])
            # display price and format price as currency
            st.write('Price:', f'{df[df["product_id"] == int(lst_product_id[i])]["price"].values[0]:,.0f} đ')
            # display rating of product and format rating as float with 2 decimal
            st.write('Rating:', f'{df[df["product_id"] == int(lst_product_id[i])]["rating"].values[0]:.2f}')
            
            st.write('------------------------------------------------------------------------------------------------------------------')
        else:
            # show image not_image.jpg to replace image N/A and format image size
            st.markdown(f'<a href="{lst_link_product[i]}" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg" width="500" height="500"></a>',
                        # get product_id
                        unsafe_allow_html=True
                        )
            
            st.button('Recommendations for this product', key = lst_product_id[i], on_click = lambda id=lst_product_id[i]: show_recommendations(product_id_new=id))
                                    
            # display title of product
            st.write(lst_title[i])
            # display price and format price as currency
            st.write('Price:', f'{df[df["product_id"] == int(lst_product_id[i])]["price"].values[0]:,.0f} đ')
            # display rating of product and format rating as float with 2 decimal
            st.write('Rating:', f'{df[df["product_id"] == int(lst_product_id[i])]["rating"].values[0]:.2f}')
            
            st.write('------------------------------------------------------------------------------------------------------------------')
            
