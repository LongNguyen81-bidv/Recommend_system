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

from function import *

# load data
df1 = pd.read_csv('Products_1.csv')
df2 = pd.read_csv('Products_2.csv')

# combine 2 dataframes
df = pd.concat([df1, df2], ignore_index=True)

# define a function to get recommendations for a user return product_id and convert to list
def get_recommendations_list(user):
    # get the user_id_idx for the user_id
    user_id_idx = df_user_user_id.filter(df_user_user_id.user == user).collect()[0][0]
    # get the recommendations
    result = spark.read.parquet('user_recs.parquet')
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


##### Build app

# Title
st.image('download.jpg')

st.title("Trung tâm tin học - ĐH KHTN")
st.header('Data Science and Machine Learning Certificate')
st.markdown('#### Project: Recommendation System on Shopee')

# st.video('https://www.youtube.com/watch?v=q3nSSZNOg38&list=PLFTWPHJsZXVVnckL0b3DYmHjjPiRB5mqX')

menu = ['Overview', 'Recommendation']
# choice = st.radio('Menu', menu)
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

    
elif choice == 'Recommendation':
    st.subheader('Recommendation')
    menu_2 = ['Recommend from keywords', 'Recommend from product_id','Recommend from user']
    choice_2 = st.radio('Menu', menu_2)
    
    if choice_2 == 'Recommend from keywords':
            
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
        
        
        
    elif choice_2 == 'Recommend from product_id':
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
    elif choice_2 == 'Recommend from user':
        st.write('''
        Input a user and the model will recommend you 5 products
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

                # list of product_id
                product_id_list = get_recommendations_list(user)

                # turn off spark
                spark.stop()
                # stop spark context
                sc.stop()
                
                                
                # show data with top 10 similar products_id select more information from df
                st.table(df[df['product_id'].isin(product_id_list)][['product_id','product_name','sub_category','price','rating']])
                
                
            else:
                st.write('Please input a user_id')