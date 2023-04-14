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
# import random library
import random

from function import *

# load data
df1 = pd.read_csv('Products_1.csv')
df2 = pd.read_csv('Products_2.csv')

# combine 2 dataframes
df = pd.concat([df1, df2], ignore_index=True)

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
        - Lựa chọn thuật toán Gensim cho bài toán Contentbased filtering
        - Áp dụng thuật toán ALS cho bài toán Collaborative filtering
    ''')
    st.subheader('2.Giáo viên hướng dẫn')
    st.write('''
    **Cô : Khuất Thùy Phương**
    ''')
    st.subheader('3.Học viên thực hiện')
    st.write('''
    **HV : Thái Thanh Phong - Nguyễn Hoàng Long**
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
                st.table(df[df['product_id'].isin(sim_products(keywords))][['product_id','product_name','price','rating']])
                
                # get image and title of top 10 similar products_id
                lst_link = df[df['product_id'].isin(sim_products(keywords))]['image'].tolist()
                
                # get index of element in lst_link != N/A
                lst_index = [i for i, x in enumerate(lst_link) if str(x) != 'nan']
                # drop N/A value in lst_link
                lst_link = [x for x in lst_link if str(x) != 'nan']
                
                # get list title of lst_link = product_name     
                lst_title = df[df['product_id'].isin(sim_products(keywords))]['product_name'].tolist()
                # get title of lst_link = product_name from list lst_title with index in lst_index
                lst_title = [lst_title[i] for i in lst_index]
                
                # get list product_id of lst_link = product_id
                lst_product_id = df[df['product_id'].isin(sim_products(keywords))]['product_id'].tolist()
                # get product_id of lst_link = product_id from list lst_product_id with index in lst_index
                lst_product_id = [str(lst_product_id[i]) for i in lst_index]
                
                # concat lst_title and lst_product_id
                lst_title = [lst_title[i] + ' - ' + lst_product_id[i] for i in range(len(lst_title))]
                
                st.image(   lst_link, 
                            width=150, 
                            caption=lst_title,
                                                        )
                
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
                st.table(df[df['product_id'].isin(sim_products_tfidf_gensim(product_id))][['product_id','product_name','price','rating']])
                
                # get image and title of top 10 similar products_id
                lst_link = df[df['product_id'].isin(sim_products_tfidf_gensim(product_id))]['image'].tolist()
                
                # get index of element in lst_link != N/A
                lst_index = [i for i, x in enumerate(lst_link) if str(x) != 'nan']
                # drop N/A value in lst_link
                lst_link = [x for x in lst_link if str(x) != 'nan']
                
                # get list title of lst_link = product_name     
                lst_title = df[df['product_id'].isin(sim_products_tfidf_gensim(product_id))]['product_name'].tolist()
                # get title of lst_link = product_name from list lst_title with index in lst_index
                lst_title = [lst_title[i] for i in lst_index]
                
                # get list product_id of lst_link = product_id
                lst_product_id = df[df['product_id'].isin(sim_products_tfidf_gensim(product_id))]['product_id'].tolist()
                # get product_id of lst_link = product_id from list lst_product_id with index in lst_index
                lst_product_id = [str(lst_product_id[i]) for i in lst_index]
                
                # concat lst_title and lst_product_id
                lst_title = [lst_title[i] + ' - ' + lst_product_id[i] for i in range(len(lst_title))]
                
                st.image(   lst_link, 
                            width=150, 
                            caption=lst_title,
                                                        )
                
            else:
                st.write('Please input a product_id')
    elif choice_2 == 'Recommend from user':
        st.write('''
        Input a user and the model will recommend you 5 products
        ''')
        
        user = st.text_input('Input a user')
        
        # load df_user_user_id.parquet use pandas
        df_user_user_id = pd.read_parquet('df_user_user_id.parquet')
        
        df_user_unique = pd.Series(df_user_user_id["user"].unique())
        user_rand_lst = [random.randint(0, df_user_unique.shape[0]) for i in range(30)] 
        user_random = df_user_unique.iloc[user_rand_lst].tolist()
        user_selected = st.selectbox("Chọn user ID", options = user_random)
        
        
        if st.button('Recommend from user'):
            if user != '':
                
                # list of product_id
                product_id_list = get_recommendations_list(user)
                                                
                # show data with top 10 similar products_id select more information from df
                st.table(df[df['product_id'].isin(product_id_list)][['product_id','product_name','price','rating']])
                
                # get image and title of top 10 similar products_id
                lst_link = df[df['product_id'].isin(product_id_list)]['image'].tolist()
                
                # get index of element in lst_link != N/A
                lst_index = [i for i, x in enumerate(lst_link) if str(x) != 'nan']
                # drop N/A value in lst_link
                lst_link = [x for x in lst_link if str(x) != 'nan']
                
                # get list title of lst_link = product_name     
                lst_title = df[df['product_id'].isin(product_id_list)]['product_name'].tolist()
                # get title of lst_link = product_name from list lst_title with index in lst_index
                lst_title = [lst_title[i] for i in lst_index]
                
                # get list product_id of lst_link = product_id
                lst_product_id = df[df['product_id'].isin(product_id_list)]['product_id'].tolist()
                # get product_id of lst_link = product_id from list lst_product_id with index in lst_index
                lst_product_id = [str(lst_product_id[i]) for i in lst_index]
                
                # concat lst_title and lst_product_id
                lst_title = [lst_title[i] + ' - ' + lst_product_id[i] for i in range(len(lst_title))]
                
                st.image(   lst_link, 
                            width=150, 
                            caption=lst_title,
                                                        )
            
            elif user_selected != '':
                # list of product_id
                product_id_list = get_recommendations_list(user_selected)
                                                
                # show data with top 10 similar products_id select more information from df
                st.table(df[df['product_id'].isin(product_id_list)][['product_id','product_name','price','rating']])
                
                # get image and title of top 10 similar products_id
                lst_link = df[df['product_id'].isin(product_id_list)]['image'].tolist()
                
                # get index of element in lst_link != N/A
                lst_index = [i for i, x in enumerate(lst_link) if str(x) != 'nan']
                # drop N/A value in lst_link
                lst_link = [x for x in lst_link if str(x) != 'nan']
                
                # get list title of lst_link = product_name     
                lst_title = df[df['product_id'].isin(product_id_list)]['product_name'].tolist()
                # get title of lst_link = product_name from list lst_title with index in lst_index
                lst_title = [lst_title[i] for i in lst_index]
                
                # get list product_id of lst_link = product_id
                lst_product_id = df[df['product_id'].isin(product_id_list)]['product_id'].tolist()
                # get product_id of lst_link = product_id from list lst_product_id with index in lst_index
                lst_product_id = [str(lst_product_id[i]) for i in lst_index]
                
                # concat lst_title and lst_product_id
                lst_title = [lst_title[i] + ' - ' + lst_product_id[i] for i in range(len(lst_title))]
                
                st.image(   lst_link, 
                            width=150, 
                            caption=lst_title,
                                                        )
                                
            else:
                st.write('Please input a user_id')