# setting up the environment
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt


lr = LogisticRegression(random_state=1, max_iter=400)


file_path = "path/to/the/train/csv"

# loading the train data set
colnames = ['sentiment', 'review']
train = pd.read_csv(file_path, header=None, names=colnames)
print(train.head())

# inspecting the dataframe
print(train.info())

# setting up X and Y train
x_train = train['review']
y_train = train['sentiment']

# feature extraction using countvectorizer
cvec = CountVectorizer()
x_train_cvec = cvec.fit_transform(x_train)

score = cross_val_score(lr, x_train_cvec, y_train, cv=3)
print('Count Vectorizer Score :', score.mean())


# building the accuracy dataframe
accuracy_list = []
accuracy_list.append(score.mean())

accuracy_df = pd.DataFrame()
accuracy_df['params'] = ['CountVectorizer']
accuracy_df['score'] = accuracy_list
print(accuracy_df)

best_parameter = pd.DataFrame()
best_parameter['params'] = ['CountVectorizer']
best_parameter['score'] = accuracy_list
print(best_parameter)

# tuning the hyperparameters

# NGRAMS


def count_vec_ngram(params, X_train, Y_train):
    cvec_p = CountVectorizer(ngram_range=(params))
    xtrain_cvec_p = cvec_p.fit_transform(X_train)
    cvec_score_p = cross_val_score(
        lr, xtrain_cvec_p, Y_train, cv=3, error_score='raise')
    return cvec_score_p.mean()


params = [(1, 1), (1, 2), (1, 3), (1, 4)]
ngram_scores = []

for p in params:
    ngram_scores.append(count_vec_ngram(p, x_train, y_train))

ngrams = ['cvec gram_1', 'cvec gram_2', 'cvec gram_3', 'cvec gram_4']
ngram_df = pd.DataFrame(
    {'params': ngrams, 'score': ngram_scores}, index=[0, 1, 2, 3])
# obtaining the bestparameter
best_parameter = best_parameter.append(
    ngram_df[ngram_df['score'] == ngram_df['score'].max()])
# adding cvec score with default parameters
ngram_df = ngram_df.append(accuracy_df.iloc[:, :])


# plotting scores on the graph
sns.pointplot(x='params', y='score', data=ngram_df)
plt.ylabel('Accuracy Score')
plt.xlabel('ngrams')
plt.xticks(rotation=40)
plt.title('Accuracy of ngram range')
plt.show()

# MAX FEATURES


def count_vec_max_features(params, x_train, y_train):
    cvec_p = CountVectorizer(max_features=params)
    xtrain_cvec_p = cvec_p.fit_transform(x_train)
    cvec_score_p = cross_val_score(
        lr, xtrain_cvec_p, y_train, cv=3, error_score='raise')
    return cvec_score_p.mean()


mf_params = [None, 500, 1000, 2000, 5000]
max_feature_scores = [count_vec_max_features(
    p, x_train, y_train) for p in mf_params]
max_features = ['max_f_' + str(p) for p in mf_params]

# dataframe for scores
max_feature_df = pd.DataFrame(
    {'params': max_features, 'score': max_feature_scores}, index=[0, 1, 2, 3, 4])

# obtaining the bestparameter
best_parameter = best_parameter.append(
    max_feature_df[max_feature_df['score'] == max_feature_df['score'].max()])
# adding cvec score with default parameters
max_feature_df = max_feature_df.append(accuracy_df.iloc[:, :])

# print(best_parameter)
print(max_feature_df)
print(best_parameter)
# visualizing the results
sns.pointplot(x='params', y='score', data=max_feature_df)
plt.ylabel('Accuracy Score')
plt.xlabel('Max Features')
plt.xticks(rotation=40)
plt.title('Max Features')
plt.show()

# MAX DF
# corpus specific stopwords


def count_vec_max_df(params, x_train, y_train):
    cvec_p = CountVectorizer(max_df=params)
    xtrain_cvec_p = cvec_p.fit_transform(x_train)
    cvec_score_p = cross_val_score(
        lr, xtrain_cvec_p, y_train, cv=3, error_score='raise')
    return cvec_score_p.mean()


maxdf_params = [0.25, 0.50, 0.75, 1.00]
max_df_scores = [count_vec_max_df(
    p, x_train, y_train) for p in maxdf_params]
max_df = ['max_f_' + str(p) for p in maxdf_params]

# dataframe for scores
max_df_df = pd.DataFrame(
    {'params': max_df, 'score': max_df_scores}, index=[0, 1, 2, 3])
# obtaining the bestparameter
best_parameter = best_parameter.append(
    max_df_df[max_df_df['score'] == max_df_df['score'].max()])
# adding cvec score with default parameters
max_df_df = max_df_df.append(accuracy_df.iloc[:, :])

# visualizing the results
sns.pointplot(x='params', y='score', data=max_df_df)
plt.ylabel('Accuracy Score')
plt.xlabel('max_df')
plt.xticks(rotation=40)
plt.title('max df accuracy')
plt.show()


# combining highest parameters
sns.pointplot(x='params', y='score', data=best_parameter)
plt.ylabel('Accuracy Score')
plt.xlabel('Parameters')
plt.xticks(rotation=40)
plt.title('Comparison of Highest Parameters')
plt.show()

# combining the parameters
cvec = CountVectorizer(ngram_range=(1, 2), max_df=0.5)
# fitting the training_model
x_train_cvec = cvec.fit_transform(x_train)
# cross validating the score
cvec_score = cross_val_score(lr, x_train_cvec, y_train, cv=3)
print(cvec_score.mean())

# using only one parameter
cvec = CountVectorizer(ngram_range=(1, 2))
# fitting the training_model
x_train_cvec = cvec.fit_transform(x_train)
# cross validating the score
cvec_score = cross_val_score(lr, x_train_cvec, y_train, cv=3)
print(cvec_score.mean())
