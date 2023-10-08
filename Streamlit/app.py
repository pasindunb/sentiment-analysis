import streamlit as st
import pandas as pd
import string
import nltk
import re
nltk.download('stopwords')
nltk.download("vader_lexicon")
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from io import StringIO
from streamlit.runtime.state import session_state
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
import plotly.express as px
import emoji
import calendar
from streamlit_option_menu import option_menu


@st.cache_data(persist=True)
def load_data():
    data = pd.read_csv("drive/MyDrive/Chat_Analysis/Streamlit/Preprocessed_tweet.csv",low_memory=True, usecols=[*range(1,10)])
    data['Date']=pd.to_datetime(data['Date'],errors='coerce')
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    return data


def date_range():
    st.header("Filter by Date")
    start = st.date_input("Start Date:- (Please input on or after 2008-05-08)",pd.to_datetime("2014-01-01",format="%Y-%m-%d"))
    end = st.date_input("End Date:- (Please input on or before 2017-12-03)",pd.to_datetime("2014-12-31",format="%Y-%m-%d"))
    
    return start,end;


def preprocess(text):
    text = text.lower()
    text = re.sub('http://\S+|https://\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join(i for i in text if not i.isdigit())
    text = emoji.demojize(text,delimiters=("",""))
    text = text.replace('_',' ').replace('-',' ')

    stopWord = nltk.corpus.stopwords.words('english')
    text = [word for word in text.split() if word not in stopWord]
    text = ' '.join(text)
    
    return text


def wordcloud(text,title):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    text = WordCloud().generate(str(text))
    plt.imshow(text)
    plt.axis('off')
    plt.title(title)
    st.pyplot()


def sentimentGenerator(text):
    analyzer = SentimentIntensityAnalyzer()
    result = analyzer.polarity_scores(text)

    if result['compound'] > 0.35:
      st.success("Sentiment is Positive")
    elif result['compound'] < (-0.25):
      st.error("Sentiment is Negative")
    else:
      st.info("Sentiment is Neutral")
    
    st.write(f"Positive - {round(result['pos']*100,2)}%")
    st.write(f"Neutral - {round(result['neu']*100,2)}%")
    st.write(f"Negative - {round(result['neg']*100,2)}%")

def DownloadDataFrame(df,fileName):
    download = st.download_button(label="Download data as CSV", data=ExtractedData.to_csv().encode('utf-8'),
                       file_name=fileName, mime='text/csv',)
    if download:
      st.success("DataFrame saved as an .csv file")


st.set_page_config(page_title="Chat Analysis",page_icon="🕵️‍♂️",layout="wide",initial_sidebar_state="expanded")
#------------------------------------------------------- Main Menu -------------------------------------------------------

selected = option_menu(
  menu_title=None,
  options=["Home","EDA","Sentiment Generator"],
  icons=['house','bar-chart','emoji-heart-eyes'],
  menu_icon = 'cast',
  orientation='horizontal',
  styles={
    "icon":{"color":"red","font-size":"25px"},
    "nav-link":{"font-size":"25px","--hover-color":"#417C76",},
    "nav-link-selected":{"background-color":"#0c7c72"}
  },
)

#------------------------------------------------------- Home Page -------------------------------------------------------

if selected == "Home":
  st.title("Sentiment Analysis for Customer Support Data on Twitter")
  st.write("""Here is the streamlit dashboard to display sentiment analysis of customer support data on twitter.\n
  **Target Data Set** : Customer Support Data on Twitter - 2.8 million of data\n
  **Description of data :**\n
  **tweet_id** : A unique, anonymized ID for the Tweet. Referenced by response_tweet_id and in_response_to_tweet_id.\n
  **Author_id** : A unique, anonymized user ID. @s in the dataset have been replaced with their associated anonymized user ID.\n
  **Inbound** : Whether the tweet is "inbound" to a company doing customer support on Twitter. This feature is useful when re-organizing data for training conversational models.\n
  **Created_at** : Date and time when the tweet was sent.\n
  **Text** : Tweet content. Sensitive information like phone numbers and email addresses are replaced with mask values like email.\n
  ***Response_tweet_id*** : IDs of tweets that are responses to this tweet, comma-separated.\n
  ***In_response_to_tweet_id*** : ID of the tweet this tweet is in response to, if any.""")

  st.markdown("## Overview of Sentiments for Customer Support Data on Twitter")
  st.write('''<span style="font-family: cursive; font-size: 3rem; color: green">**POSITIVE 28%**&nbsp;&nbsp;&nbsp;</span>
  <span style="font-family: cursive; font-size: 3rem; color: blue">**NEUTRAL 39%**&nbsp;&nbsp;&nbsp;</span>
  <span style="font-family: cursive; font-size: 3rem; color: red">**NEGATIVE 33%**</span>''', unsafe_allow_html = True)
  
#------------------------------------------------------- EDA Page -------------------------------------------------------

elif selected == "EDA":
  data = load_data()
  start,end = date_range()

  extract = st.button('Extract data')
  if st.session_state.get('button') != True:
    st.session_state['button'] = extract

  if st.session_state['button'] == True:
    date_range_df = data.loc[data["Date"].between(str(start), str(end))]
    st.header("Extracted Data set")
    st.dataframe(date_range_df)
    ExtractedData = date_range_df
    fileName = 'Date Range ([%s] - [%s]).csv'%(str(start), str(end))
    DownloadDataFrame(ExtractedData,fileName)

    #-------------------------------------- Filter by Author --------------------------------------

    st.sidebar.header("Filter by Author")
    author_list = date_range_df['Author_ID'].value_counts().index.tolist()
    author = st.sidebar.selectbox("Select Author :",['All'] + author_list)
    
    if author != 'All':
      date_range_df = date_range_df[date_range_df['Author_ID']==author]
      st.header("Filterd Data set")
      st.subheader(f"Author :- {author}")
      st.dataframe(date_range_df)
      fileName = '%s ([%s] - [%s]).csv'%(author,str(start), str(end))
      DownloadDataFrame(date_range_df,fileName)
    else:
      author = "All Author"
    #-------------------------------------- WordCloud for Sentiments --------------------------------------

    st.sidebar.header("Word Cloud")
    word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', (None,'All','Positive', 'Neutral', 'Negative'),key='1')
    if word_sentiment != None:
      try:
        if word_sentiment == 'All':
          st.subheader("Word Cloud for All sentiments")
          text = date_range_df['Messege'].tolist()
          text = [str(x) for x in text]
          text = ' '.join(text)
          title = "Word cloud for All sentiments"
          wordcloud(text,title)
        else:
          st.subheader(f"Word cloud for {author}'s {word_sentiment} sentiments")
          text = date_range_df[date_range_df['NLTK_Tag']==word_sentiment]['Messege'].tolist()
          text = [str(x) for x in text]
          text = ' '.join(text)
          title = "Word cloud for %s's %s sentiments" % (author,word_sentiment)
          wordcloud(text,title)
      except:
        st.error(f"There is no {word_sentiment} sentiment tweets on {author}'s tweets")

    #-------------------------------------- Draw a Bar and Pie Chart --------------------------------------

    st.sidebar.header("Bar Chart/Pie Chart")
    select = st.sidebar.radio('What visualization type do you want to display number of sentiments ?', (None,'Bar Chart', 'Pie Chart'))
    if select != None:
      sentiment = date_range_df['NLTK_Tag'].value_counts().index.tolist()
      sentiment_count = date_range_df['NLTK_Tag'].value_counts().tolist()
      percentage = [i*100/sum(sentiment_count) for i in sentiment_count]
      percentage = [str(round(i,2))+'%' for i in percentage]
      sentiment_count = pd.DataFrame({'Sentiment':sentiment, 'Tweets':sentiment_count})
      
      st.markdown("### Number of tweets by sentiment")
      st.subheader(f"Author :- {author}")
      if select == 'Bar Chart':
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets',text = percentage, color='Sentiment')
        st.plotly_chart(fig)
      else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)

    #-------------------------------------- line chart --------------------------------------

    st.sidebar.header("Line Chart")
    year = st.sidebar.selectbox("What year do you want to see the sentiment changes monthly ?",[None]+list(range(2008, 2018)))
    if year != None:
      df = data[['Date','NLTK_Tag']]
      df['Date'] = pd.to_datetime(df['Date'],errors='coerce')
      df['Year'] = df['Date'].dt.year
      df['Month'] = df['Date'].dt.month
      df = df[df['Year']==year]
      
      pos,neg,neu=[],[],[]
      for month in range(1,13):
        df0 = df[df['Month']==month]
        pos_cnt,neu_cnt,neg_cnt = 0,0,0
        for sntmnt in df0['NLTK_Tag'].tolist():
          if sntmnt == 'Positive':
            pos_cnt += 1
          elif sntmnt == 'Neutral':
            neu_cnt += 1
          else:
            neg_cnt += 1
        pos.append(pos_cnt)
        neu.append(neu_cnt)
        neg.append(neg_cnt)
      
      line_chart_data = pd.DataFrame({'Positive':pos,'Neutral':neu,'Negative':neg},index = list(calendar.month_name)[1:])
      st.markdown("### Monthly Changes of Sentiments Customer Support Data over Year")
      fig = px.line(line_chart_data,color_discrete_map={"Positive": "green","Neutral": "white","Negative": "red"}).update_layout(
        title = {'text':f"Year - {year}",'x':0.5}, xaxis_title="Month", yaxis_title="Number of Tweets",legend_title="Sentiment")
      
      st.plotly_chart(fig, use_container_width=True)

#------------------------------------------------------- Sentiment Generator Page -------------------------------------------------------

elif selected == "Sentiment Generator":
  st.title("Sentiment Generator")

  option = st.radio("Select your input option :",("Type a text",".txt"))

  #-------------------------------------- Input as a text and .txt file--------------------------------------

  if option == "Type a text":
    text = st.text_input("Please enter your text in ***english*** for analysis :")
  else:
    file = st.file_uploader("Choose a file : ")
    if file != None:
      stringio = StringIO(file.getvalue().decode("utf-8"))
      text = stringio.read()

  #-------------------------------------- Generate sentiment and wordCloud --------------------------------------

  generate = st.button('Generate')
  if st.session_state.get('generate') != True:
      st.session_state['generate'] = generate
  if st.session_state['generate']:
    text = preprocess(text)
    sentimentGenerator(text)
    st.markdown("### Do You Want to Draw a WordCloud for Generated Text?")
    check = st.checkbox("Draw a wordcloud")
    if check:
      title = "WordCloud for Generated Text"
      wordcloud(text,title)
