import streamlit as st
import pandas as pd
import numpy as np
import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Reviews Summarizer")

col1, col2 = st.columns(2, gap="large")

# df = pd.read_csv('nyka_top_brands_cosmetics_product_reviews.csv')
df = pd.read_csv('https://raw.githubusercontent.com/DarrenTeo/Review_Summarizer/main/nyka_top_brands_cosmetics_product_reviews.csv')

# product_df = pd.read_csv('nyka_popular_brands_products_2022_10_16.csv')
product_df = pd.read_csv('https://raw.githubusercontent.com/DarrenTeo/Review_Summarizer/main/nyka_popular_brands_products_2022_10_16.csv')

fact_df = pd.merge(df, product_df[['product_id', 'image_url']], how='left', on=['product_id'])

#################  
### Add State ###   
#################
if 'fact_df' not in st.session_state:
    st.session_state.fact_df=fact_df

###############
### Filters ###
###############

with st.sidebar:
    with st.expander('Filters', expanded=True):
        #############
        ### Brand ###
        #############
        brand_list = fact_df['brand_name'].dropna().sort_values(ascending=True).unique()
        all_brand = st.checkbox("Select All Brands",key='brand_checkbox')
        brand_container = st.container()

        if all_brand:
            selected_brand = brand_container.multiselect("Select Brand",
                 brand_list,brand_list)
        else:
            selected_brand =  brand_container.multiselect("Select Brand",
                brand_list,default="L'Oreal Paris")

    # with st.expander('Product Filter'):
        ##################
        ### Product ID ###
        ##################

        product_list = fact_df.loc[df['brand_name'].isin(selected_brand)]['product_id'].dropna().sort_values(ascending=True).unique()
        all_product = st.checkbox("Select All Products",key='product_checkbox')
        product_container = st.container()

        if all_product:
            selected_product = product_container.multiselect("Select Product",
                 product_list,product_list)
        else:
            selected_product =  product_container.multiselect("Select Product",
                product_list,default=479)
            
with col1:
        
    ########################
    ### DF after Filters ###
    ########################
    st.session_state['fact_df'] = fact_df[(fact_df.brand_name.isin(selected_brand))&
                                          (fact_df.product_id.isin(selected_product))
                                         ]

    st.subheader('Selected Product')
    st.session_state['fact_df']['product_title'].iloc[0]

    st.subheader('Preview of Data')
    st.dataframe(data=st.session_state['fact_df'].head(), use_container_width=True)

    st.subheader('Preview of Selected Reviews ')
    total_reviews = len(st.session_state.fact_df['review_text'])
    number_of_reviews = st.slider("Review", min_value=0, max_value=total_reviews, value=[0,100], step=1)

    st.write("Total Reviews Selected",number_of_reviews[1]-number_of_reviews[0])


    documents = st.session_state.fact_df['review_text'][number_of_reviews[0]:number_of_reviews[1]]
    st.dataframe(data=documents, use_container_width=True)

    number_of_doc = len(st.session_state['fact_df']['review_text'])
    raw_data = " ".join(documents)

    # st.subheader('Preview of Combined Reviews (500 characters)')
    # raw_data[0:500]

    def clean_data(data):
      text = re.sub(r"\[[0-9]*\]"," ",data)
      text = text.lower()
      text = re.sub(r'\s+'," ",text)
      text = re.sub(r","," ",text)
      return text
    cleaned_article_content = clean_data(raw_data)
    # cleaned_article_content[0:100]
    
    
with col2:
    st.subheader('Summarized Reviews')
    # st.markdown('---')
    # For Strings
    parser = PlaintextParser.from_string(cleaned_article_content,Tokenizer("english"))

    summarizer = LexRankSummarizer()
    #Summarize the document with 2 sentences
    summary = summarizer(parser.document, 2)
    
    
    for sentence in summary:
        st.subheader(sentence)

    st.markdown('---')
    st.subheader('WordCloud')
    

    # Create and generate a word cloud image:
    wordcloud = WordCloud().generate(cleaned_article_content)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    # plt.show()
    st.pyplot(plt)
    
with st.sidebar:
    st.write("Data Source - https://www.kaggle.com/datasets/jithinanievarghese/cosmetics-and-beauty-products-reviews-top-brands")
