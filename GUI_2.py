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

def show_recommendations(product_id_new=100):
    
    # get image and title of top 10 similar products_id_new
    lst_link = df[df['product_id'].isin(sim_products_tfidf_gensim(product_id_new))]['image'].tolist()
    
    # get list title of lst_link = product_name
    lst_title = df[df['product_id'].isin(sim_products_tfidf_gensim(product_id_new))]['product_name'].tolist()
    
    # get list product_id of lst_link = product_id
    lst_product_id = df[df['product_id'].isin(sim_products_tfidf_gensim(product_id_new))]['product_id'].tolist()
    
    # concat lst_title and lst_product_id
    lst_title = [lst_title[i] + '  /  Product ID: ' + str(lst_product_id[i]) for i in range(len(lst_title))]
    
    # show image and title of top 10 similar products_id_new
    
    for i in range(len(lst_link)):
        if str(lst_link[i]) != 'nan':
            # display image of product and get product_id, after click on image, it will get product_id and assign value to product_id_new variable
            
            # Define a callback function that will be executed when the button is clicked
            # def on_image_click():
            #     product_id_new = lst_product_id[i]
            #     return product_id_new
            
            st.image(lst_link[i], width = 200, use_column_width = True)
            # st.button('Click here to choose product', key=image, on_click=show_recommendations(on_click_image))
            
            
            # st.markdown(f'<a href="{lst_product_id[i]}" target="_blank"><img src="{lst_link[i]}" width="500" height="500"></a>',
            #             # get product_id
            #             unsafe_allow_html=True
            #             )
            # display title of product
            st.write(lst_title[i])
            # display price and format price as currency
            st.write('Price:', f'{df[df["product_id"] == int(lst_product_id[i])]["price"].values[0]:,.0f} đ')
            # display rating of product and format rating as float with 2 decimal
            st.write('Rating:', f'{df[df["product_id"] == int(lst_product_id[i])]["rating"].values[0]:.2f}')
            
            st.write('------------------------------------------------------------------------------------------------------------------')
        else:
            # use image not_image.jpg to replace image N/A
            st.image('No-image-available.jpg', width = 200, use_column_width = True)
            
            
            
            # st.markdown(f'<a href="{lst_product_id[i]}" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg" width="500" height="500"></a>',
            #             # get product_id
            #             unsafe_allow_html=True
            #             )
            # display title of product
            st.write(lst_title[i])
            # display price and format price as currency
            st.write('Price:', f'{df[df["product_id"] == int(lst_product_id[i])]["price"].values[0]:,.0f} đ')
            # display rating of product and format rating as float with 2 decimal
            st.write('Rating:', f'{df[df["product_id"] == int(lst_product_id[i])]["rating"].values[0]:.2f}')
            
            st.write('------------------------------------------------------------------------------------------------------------------')
        
    
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
                                
                # get image and title of top 10 similar products_id
                lst_link_image = df[df['product_id'].isin(sim_products(keywords))]['image'].tolist()
                
                # get list link of product = link
                lst_link_product = df[df['product_id'].isin(sim_products(keywords))]['link'].tolist()
                                
                # get list title of lst_link = product_name     
                lst_title = df[df['product_id'].isin(sim_products(keywords))]['product_name'].tolist()
                                
                # get list product_id of lst_link = product_id
                lst_product_id = df[df['product_id'].isin(sim_products(keywords))]['product_id'].tolist()
                                
                # concat lst_title and lst_product_id
                lst_title = [lst_title[i] + '  /  Product_id: ' + str(lst_product_id[i]) for i in range(len(lst_title))]
                
                            
                for i in range(len(lst_link_image)):
                    if str(lst_link_image[i]) != 'nan':
                        st.markdown(f'<a href="{lst_link_product[i]}" target="_blank"><img src="{lst_link_image[i]}" width="500" height="500"></a>',
                                    # get product_id
                                    unsafe_allow_html=True
                                    )
                        # display title of product
                        st.write(lst_title[i])
                        # display price and format price as currency
                        st.write('Price:', f'{df[df["product_id"] == int(lst_product_id[i])]["price"].values[0]:,.0f} đ')
                        # display rating of product and format rating as float with 2 decimal
                        st.write('Rating:', f'{df[df["product_id"] == int(lst_product_id[i])]["rating"].values[0]:.2f}')
                        
                        st.write('------------------------------------------------------------------------------------------------------------------')
                    else:
                        # use image not_image.jpg to replace image N/A
                        # st.image('download.jpg')
                        st.markdown(f'<a href="{lst_link_product[i]}" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg" width="500" height="500"></a>',
                                    # get product_id
                                    unsafe_allow_html=True
                                    )
                        # display title of product
                        st.write(lst_title[i])
                        # display price and format price as currency
                        st.write('Price:', f'{df[df["product_id"] == int(lst_product_id[i])]["price"].values[0]:,.0f} đ')
                        # display rating of product and format rating as float with 2 decimal
                        st.write('Rating:', f'{df[df["product_id"] == int(lst_product_id[i])]["rating"].values[0]:.2f}')
                        
                        st.write('------------------------------------------------------------------------------------------------------------------')
                    

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
                # get image and title of top 10 similar products_id
                lst_link_image = df[df['product_id'].isin(sim_products_tfidf_gensim(product_id))]['image'].tolist()
                
                # get list product_id of lst_link = product_id
                lst_link_product = df[df['product_id'].isin(sim_products_tfidf_gensim(product_id))]['link'].tolist()
                
                # get list title of lst_link = product_name
                lst_title = df[df['product_id'].isin(sim_products_tfidf_gensim(product_id))]['product_name'].tolist()
                
                # get list product_id of lst_link = product_id
                lst_product_id = df[df['product_id'].isin(sim_products_tfidf_gensim(product_id))]['product_id'].tolist()
                
                # concat lst_title and lst_product_id
                lst_title = [lst_title[i] + '  /  Product_id: ' + str(lst_product_id[i]) for i in range(len(lst_title))]
                
                for i in range(len(lst_link_image)):
                    if str(lst_link_image[i]) != 'nan':
                        st.markdown(f'<a href="{lst_link_product[i]}" target="_blank"><img src="{lst_link_image[i]}" width="500" height="500"></a>',
                                    # get product_id
                                    unsafe_allow_html=True
                                    )
                        # display title of product
                        st.write(lst_title[i])
                        # display price and format price as currency
                        st.write('Price:', f'{df[df["product_id"] == int(lst_product_id[i])]["price"].values[0]:,.0f} đ')
                        # display rating of product and format rating as float with 2 decimal
                        st.write('Rating:', f'{df[df["product_id"] == int(lst_product_id[i])]["rating"].values[0]:.2f}')
                        
                        st.write('------------------------------------------------------------------------------------------------------------------')
                    else:
                        # use image not_image.jpg to replace image N/A
                        # st.image('download.jpg')
                        st.markdown(f'<a href="{lst_link_product[i]}" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg" width="500" height="500"></a>',
                                    # get product_id
                                    unsafe_allow_html=True
                                    )
                        # display title of product
                        st.write(lst_title[i])
                        # display price and format price as currency
                        st.write('Price:', f'{df[df["product_id"] == int(lst_product_id[i])]["price"].values[0]:,.0f} đ')
                        # display rating of product and format rating as float with 2 decimal
                        st.write('Rating:', f'{df[df["product_id"] == int(lst_product_id[i])]["rating"].values[0]:.2f}')
                        
                        st.write('------------------------------------------------------------------------------------------------------------------')                                
                
                
            else:
                st.write('Please input a product_id')
    elif choice_2 == 'Recommend from user':
        st.write('''
        Input a user and the model will recommend you 5 products
        ''')
        
        user = st.text_input('Input a user')
            
        user_ = df_new_user_recs['user'].tolist()
        
        
        # creat list of user to show in selectbox get 50 user first
        user_lst = df_new_user_recs['user'].tolist()[:100]
        # creat selectbox to choose user
        user_selected = st.selectbox("Chọn user ID", options = user_lst)
        
        if st.button('Recommend from user'):
            # if user != '':
            if (user != '') & (user in user_):
                
                product_id_list = get_recommendations_list(user)
                
                # get image and title of top 10 similar products_id
                lst_link_image = df[df['product_id'].isin(product_id_list)]['image'].tolist()
                
                # get link of product 
                lst_link_product = df[df['product_id'].isin(product_id_list)]['link'].tolist()
                
                # get list title of lst_link = product_name
                lst_title = df[df['product_id'].isin(product_id_list)]['product_name'].tolist()
                
                # get list product_id of lst_link = product_id
                lst_product_id = df[df['product_id'].isin(product_id_list)]['product_id'].tolist()
                
                # concat lst_title and lst_product_id
                lst_title = [lst_title[i] + '  /  Product_id: ' + str(lst_product_id[i]) for i in range(len(lst_title))]
                
                
                # display image, title, price, rating of product
                for i in range(len(lst_link_image)):
                    if str(lst_link_image[i]) != 'nan':
                        # use image as link to product of shopee                                    
                        st.markdown(f'<a href="{lst_link_product[i]}" target="_blank"><img src="{lst_link_image[i]}" width="500" height="500"></a>',
                                    # get product_id
                                    unsafe_allow_html=True
                                    )
                        # display title of product
                        st.write(lst_title[i])
                        # display price and format price as currency
                        st.write('Price:', f'{df[df["product_id"] == int(lst_product_id[i])]["price"].values[0]:,.0f} đ')
                        # display rating of product and format rating as float with 2 decimal
                        st.write('Rating:', f'{df[df["product_id"] == int(lst_product_id[i])]["rating"].values[0]:.2f}')
                        
                        st.write('------------------------------------------------------------------------------------------------------------------')
                    else:
                        # use image not_image.jpg to replace image N/A and format image size, use link to product of shopee
                        st.markdown(f'<a href="{lst_link_product[i]}" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg" width="500" height="500"></a>',
                                    # get product_id
                                    unsafe_allow_html=True
                                    )
                        # display title of product
                        st.write(lst_title[i])
                        # display price and format price as currency
                        st.write('Price:', f'{df[df["product_id"] == int(lst_product_id[i])]["price"].values[0]:,.0f} đ')
                        # display rating of product and format rating as float with 2 decimal
                        st.write('Rating:', f'{df[df["product_id"] == int(lst_product_id[i])]["rating"].values[0]:.2f}')
                        
                        st.write('------------------------------------------------------------------------------------------------------------------')
                
                            
            elif (user != '') & (user not in user_):
                st.write('User not found')
            
            elif user_selected != '':
                # list of product_id
                product_id_list = get_recommendations_list(user_selected)
                                                
                # get image and title of top 10 similar products_id
                lst_link_image = df[df['product_id'].isin(product_id_list)]['image'].tolist()
                
                # get link of product 
                lst_link_product = df[df['product_id'].isin(product_id_list)]['link'].tolist()
                
                # get list title of lst_link = product_name
                lst_title = df[df['product_id'].isin(product_id_list)]['product_name'].tolist()
                
                # get list product_id of lst_link = product_id
                lst_product_id = df[df['product_id'].isin(product_id_list)]['product_id'].tolist()
                
                # concat lst_title and lst_product_id
                lst_title = [lst_title[i] + '  /  Product_id: ' + str(lst_product_id[i]) for i in range(len(lst_title))]
                
                
                # display image, title, price, rating of product
                for i in range(len(lst_link_image)):
                    if str(lst_link_image[i]) != 'nan':
                        # use image as link to product of shopee                                    
                        st.markdown(f'<a href="{lst_link_product[i]}" target="_blank"><img src="{lst_link_image[i]}" width="500" height="500"></a>',
                                    # get product_id
                                    unsafe_allow_html=True
                                    )
                        # display title of product
                        st.write(lst_title[i])
                        # display price and format price as currency
                        st.write('Price:', f'{df[df["product_id"] == int(lst_product_id[i])]["price"].values[0]:,.0f} đ')
                        # display rating of product and format rating as float with 2 decimal
                        st.write('Rating:', f'{df[df["product_id"] == int(lst_product_id[i])]["rating"].values[0]:.2f}')
                        
                        st.write('------------------------------------------------------------------------------------------------------------------')
                    else:
                        # use image not_image.jpg to replace image N/A and format image size, use link to product of shopee
                        st.markdown(f'<a href="{lst_link_product[i]}" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg" width="500" height="500"></a>',
                                    # get product_id
                                    unsafe_allow_html=True
                                    )
                        # display title of product
                        st.write(lst_title[i])
                        # display price and format price as currency
                        st.write('Price:', f'{df[df["product_id"] == int(lst_product_id[i])]["price"].values[0]:,.0f} đ')
                        # display rating of product and format rating as float with 2 decimal
                        st.write('Rating:', f'{df[df["product_id"] == int(lst_product_id[i])]["rating"].values[0]:.2f}')
                        
                        st.write('------------------------------------------------------------------------------------------------------------------')
                    
                                
            else:
                st.write('Please input a user_id')