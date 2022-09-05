# setting up the environment
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')


lr = LogisticRegression(random_state=1, max_iter=200)

lemmatizer = WordNetLemmatizer().lemmatize
wordpunct_tokenize = WordPunctTokenizer().tokenize
en_stop = set(stopwords.words('english'))


file_path = "/the/path/to/the/labeled/training/data"

# viewing words listed as stopwords
print(en_stop)


# declaring necessary urls and setting up to web-scrape
start_url = 'https://www.imdb.com/title/tt10648342/reviews?ref_=tt_urv'
# activates the 'load more' button
load_more_url = 'https://www.imdb.com/title/tt10648342/reviews/_ajax'


params = {'ref_': 'undefined', 'paginationKey': ''}

# creating a session that allows loading of more reviews
with requests.Session() as s:
    s.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36 Edg/103.0.1264.62'
    res = s.get(start_url)

    while True:
        soup = BeautifulSoup(res.text, "html.parser")
        list_of_reviews = [reviews.get_text() for reviews in soup.find_all(
            'div', class_="text show-more__control")]

        try:
            tag = soup.find('div', class_='load-more-data')
            pagination_key = tag['data-key'].get_text()
        except AttributeError:
            break
        params['paginationKey'] = pagination_key
        res = s.get(load_more_url, params=params)


# appending data to dataframe
columns = ['review']
thor = pd.concat([pd.DataFrame([i], columns=columns)
                  for i in list_of_reviews], ignore_index=True)
print(thor.head())


# read and exploring the train data
dataset = pd.read_csv(file_path, sep='\t', index_col=None)
dataset = dataset.drop(dataset.columns[0], axis=1)
labelled_data = pd.DataFrame(dataset.loc[:14999, :])
print(labelled_data.head())
print(labelled_data.info())

# creating a function that will clean text in the dataframe
html_tags = re.compile(r'<[^<]+?>')
emojis = re.compile(
    "(["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "])"
)


def cleaning_tool(dataframe):

    dataframe['review'] = dataframe['review'].apply(
        lambda words: re.sub(html_tags, ' ', words))
    dataframe['review'] = dataframe['review'].apply(
        lambda words: re.sub(emojis, ' ', words))
    dataframe['review'] = dataframe['review'].apply(
        lambda words: ''.join([x for x in words if not x.isdigit()]))
    dataframe['review'] = dataframe['review'].apply(
        lambda words: ''.join([x for x in words if x.isascii()]))
    dataframe['review'] = dataframe['review'].apply(
        lambda words: [str(words).lower()])
    dataframe['review'] = dataframe['review'].apply(
        lambda words: wordpunct_tokenize(str(words)))
    dataframe['review'] = dataframe['review'].apply(lambda words: ''.join([str(words).translate(
        str.maketrans('', '', string.punctuation))]))
    dataframe['review'] = dataframe['review'].apply(
        lambda words: lemmatizer(str(words)))
    dataframe['review'] = dataframe['review'].apply(
        lambda words: ' '.join([x for x in str(words).split() if x not in (en_stop)]))

    return dataframe


# applying the cleaning tool to the labelled data dataframe
train = cleaning_tool(labelled_data)
print(train.head())

# setting up X and Y
Y = train['sentiment']
X = train['review']
X1 = train['review']

# converting the text into numbers
reviews_vec = CountVectorizer(ngram_range=(1, 2))
X_data = reviews_vec.fit_transform(X)
X1_data = reviews_vec.fit_transform(X1)

# creating a dataframe for use in feature comparison
X1_data_array = X1_data.astype(np.uint8).toarray()
vocab = reviews_vec.get_feature_names_out()
X1_data_df = pd.DataFrame(X1_data_array, columns=vocab)
print(X1_data_df.info())

# splitting the dataset into test, validate and train
x_train, x_temp, y_train, y_temp = train_test_split(
    X_data, Y, test_size=0.20, random_state=12)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5)

# building the model
lr.fit(x_train, y_train)

# calculating the model scores
print('validate set accuracy score: ', lr.score(x_val, y_val))
print('test set accuracy score: ', lr.score(x_test, y_test))


# PREDICTING THOR SENTIMENTS
# cleaning the dataset
thor = cleaning_tool(thor)
print(thor.head())

# vectorizing the reviews
thor_revs = thor['review']
reviews_vec = CountVectorizer(ngram_range=(1, 2))
thor_data = reviews_vec.fit_transform(thor_revs)

# creating a dataframe for feature comparison
thor_data_array = thor_data.astype(np.uint8).toarray()
thorvocab = reviews_vec.get_feature_names_out()
thor_data_df = pd.DataFrame(thor_data_array, columns=thorvocab)
print(thor_data_df.info())

# accounting for missing vocabulary
not_exist_vocab = [v for v in X1_data_df.columns.tolist()
                   if v not in thor_data_df]
thor_data_df = thor_data_df.reindex(
    columns=thor_data_df.columns.tolist() + not_exist_vocab)
print(len(not_exist_vocab))
thor_data_df = thor_data_df.fillna(0)
thor_data_df = thor_data_df[X1_data_df.columns.tolist()]
print(thor_data_df.info())


# converting back to sparse matrix
thor_data_array2 = thor_data_df.to_numpy()
thor_data = sparse.csr_matrix(thor_data_array2)


# predicting the sentiment score
predictedvalues = lr.predict(thor_data)

# Listing the number of reviews as classified by polarity
print(f'Number of Positive Reviews : {list(predictedvalues).count(1)}')
print(f'Number of Negative Reviews : {list(predictedvalues).count(0)}')
