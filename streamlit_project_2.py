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





# Part 1: Build project
# df = pd.read_csv('Products_Shopee_comments.csv')
# load data
df = pd.read_csv('Products.csv')
df['rating'] = df['rating'].astype(float).round(1)

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


# define function to get data of product_id
def get_data_product_id(product_id):
    data = df[df['product_id'] == product_id]
    return data


# define a function to get recommendations for a user return product_id and convert to list
def get_recommendations_list(user, num_recs):
    # get the user_id_idx for the user_id
    user_id_idx = df_user_user_id.filter(df_user_user_id.user == user).collect()[0][0]
    # get the recommendations
    result = model.recommendForAllUsers(num_recs)
    # get the recommendations for the user_id_idx
    result = result.filter(result['user_id_idx']==user_id_idx)
    # explode the recommendations
    result = result.select(result.user_id_idx, explode(result.recommendations))
    # get the product_id_idx and rating
    result = result.withColumn('product_id_idx', result.col.getField('product_id_idx'))\
                .withColumn('rating', result.col.getField('rating'))
    # filter the recommendations with rating >= 3.0
    result = result.filter(result.rating>=3.0)
    # join the recommendations with the product_id_product_id_idx
    result = result.join(df_product_id_product_id_idx, on=['product_id_idx'], how='left')
    # return the recommendations and convert to list
    return result.select('product_id').toPandas()['product_id'].tolist()

# Part 2: Build app

# Title
st.image('download.jpg')

st.title("Trung tâm tin học - ĐH KHTN")
st.header('Data Science and Machine Learning Certificate')
st.markdown('#### Project: Recommendation System on Shopee')

# st.video('https://www.youtube.com/watch?v=q3nSSZNOg38&list=PLFTWPHJsZXVVnckL0b3DYmHjjPiRB5mqX')

menu = ['Overview', 'Recommendation']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Overview':
    st.subheader('Overview')
    
    st.write('''
    ### PROJECT 2 : Part 1 : Data Prepreocessing

    **Mục tiêu/ Vấn đề : Xây dựng Recommendation System
    cho một hoặc một số nhóm hàng hóa trên shopee.vn giúp
    đề xuất và gợi ý cho người dùng/ khách hàng. => Xây
    dựng các mô hình đề xuất:.**
    + Content-based filtering : Hệ thống dựa trên sự tương quan của nội dung sản phẩm
    + Collaborative filtering : Hệ thống dựa trên sự tương quan của người dùng

    #### Hướng dẫn chi tiết :
    + Hiểu được vấn đề
    + Import các thư viện cần thiết và hiểu cách sử dụng
    + Đọc dữ liệu được cung cấp
    + Thực hiện EDA (Exploratory Data Analysis – Phân tích Khám phá Dữ liệu) cơ bản ( sử dụng Pandas Profifing Report )
    + Tiền xử lý dữ liệu : Làm sạch, tạo tính năng mới , lựa chọn tính năng cần thiết....
    ''')
    st.write('''
    + **Bước 1** : Business Understanding
    + **Bước 2** : Data Understanding ==> Giải quyết bài toán ..........................
    + **Bước 3** : Data Preparation/ Prepare : Chuẩn hóa tiếng việt, viết các hàm xử lý dữ liệu thô...
        
    **Xử lý tiếng việt :**
        
        **1.Tiền xử lý dữ liệu thô :**
        
        - Chuyển text về chữ thường
        - Loại bỏ các ký tự đặc biệt nếu có
        - Thay thế emojicon/ teencode bằng text tương ứng
        - Thay thế một số punctuation và number bằng khoảng trắng
        - Thay thế các từ sai chính tả bằng khoảng trắng
        - Thay thế loạt khoảng trắng bằng một khoảng trắng
        
        **2.Chuẩn hóa Unicode tiếng Việt :**
    
        **3.Tokenizer văn bản tiếng Việt bằng thư viện underthesea :**
    
        **4.Xóa các stopword tiếng Việt :** 

    + **Bước 4&5: Modeling & Evaluation/ Analyze & Report**
        - Bài toán 1: Đề xuất người dùng với Contentbased filtering
        
            - Cosine_similarity
            - Gensim
            - ......
            + Thực hiện/ đánh giá kết quả
            - Kết luận
            
        - Bài toán 2: Đề xuất người dùng với Collaborative filtering
            - Xây dựng Collaborative Filtering (pyspark.ml.recommendation.ALS)
            + Thực hiện/ đánh giá kết quả
            - RMSE
            - Kết luận
    + **Bước 6: Deployment & Feedback/ Act**
        - Xây dựng giao diện để người dùng trực tiếp nhập vào sản phẩm và gợi ý cho người dùng những sản phẩm mới.
    ''')
    st.write('''
    
    ''')
# elif choice == 'Build Model':
#     st.subheader('Build Model')
#     st.write('#### Data Preprocessing')
#     st.write('##### Show data')
#     st.table(df.head())
#     # plot bar chart for sentiment
#     st.write('##### Bar chart for sentiment')
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.bar(df['sentiment'].value_counts().index, df['sentiment'].value_counts().values)
#     ax.set_xticks(df['sentiment'].value_counts().index)
#     ax.set_xticklabels(['Positive', 'Negative'])
#     ax.set_ylabel('Number of comments')
#     ax.set_title('Bar chart for sentiment')
#     st.pyplot(fig)
    
#     # plot wordcloud for positive and negative comments
#     st.write('##### Wordcloud for positive comments')
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.imshow(WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['sentiment'] == 1]['comment'])))
#     ax.axis('off')
#     st.pyplot(fig)
    
#     st.write('##### Wordcloud for negative comments')
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.imshow(WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['sentiment'] == 0]['comment'])))
#     ax.axis('off')
#     st.pyplot(fig)
    
#     st.write('#### Build model and evaluation:')
#     st.write('##### Confusion matrix')
#     st.table(cm)
#     st.write('##### Classification report')
#     st.table(classification_report(y_test, y_pred, 
#                                     output_dict=True
#                                     ))
#     st.write('##### Accuracy')
#     # show accuracy as percentage with 2 decimal places
#     st.write(f'{accuracy_score(y_test, y_pred)*100:.2f}%')
    
elif choice == 'Recommendation':
    st.subheader('Recommendation')
    st.write('''
    Input a keywords and the model will recommend you 10 products
    ''')
    keywords = st.text_input('Input a keywords')
    if st.button('Recommend from keywords'):
        if keywords != '':
            # show data with top 10 similar products_id
            st.table(df[df['product_id'].isin(sim_products(keywords))][['product_id','product_name','sub_category','price','rating']])
        else:
            st.write('Please input a keywords')
    
    st.write('''
    Input a product_id and the model will recommend you 10 products
    ''')
    # min value of column product_id and convert to integer
    min_product_id = int(df['product_id'].min())
    # max value of column product_id and convert to integer
    max_product_id = int(df['product_id'].max())
    # format number input as integer
    product_id = st.number_input('Input a product_id', min_value=min_product_id, max_value=max_product_id, value = min_product_id, step=1)
    if st.button('Recommend from product_id'):
        if product_id != '':
            # show data of product_id
            st.write('Data of product_id:', product_id)
            st.table(df[df['product_id'] == product_id][['product_id','product_name','sub_category','price','rating']])
            # show data with top 10 similar products_id
            st.write('Top 10 similar products_id')
            st.table(df[df['product_id'].isin(sim_products_tfidf_gensim(product_id))][['product_id','product_name','sub_category','price','rating']])
        else:
            st.write('Please input a product_id')
            
    st.write('''
    Input a user and the model will recommend you 10 products
    ''')
        
    user = st.text_input('Input a user')
    if st.button('Recommend from user'):
        if user != '':
            
            import findspark
            findspark.init()
            from pyspark import SparkContext
            from pyspark.sql import SparkSession

            sc = SparkContext(master='local[*]', appName='Recommendation_ratings')
            spark = SparkSession(sc)

            from pyspark.sql.functions import explode
            
            # load df_user_user_id
            df_user_user_id = spark.read.csv('df_user_user_id.csv', inferSchema=True, header=True)
            # load df_product_id_product_id_idx
            df_product_id_product_id_idx = spark.read.csv('df_product_id_product_id_idx.csv', inferSchema=True, header=True)

            # import ALS model
            from pyspark.ml.recommendation import ALSModel
            # load model ALS
            model = ALSModel.load("als_model")
            
            # list of product_id
            product_id_list = get_recommendations_list(user, 10)
            
            # turn off spark
            spark.stop()
            # stop spark context
            sc.stop()
                        
            # show data with top 10 similar products_id select more information from df
            st.table(df[df['product_id'].isin(product_id_list)][['product_id','product_name','sub_category','price','rating']])
            
            
        else:
            st.write('Please input a user_id')