from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=33)

from sklearn.feature_extraction.text import CountVectorizer
counnt_vec = CountVectorizer()
x_count_train = counnt_vec.fit_transform(x_train)
x_count_test = counnt_vec.transform(x_test)

from sklearn.naive_bayes import  MultinomialNB
mnb_count = MultinomialNB()
mnb_count.fit(x_count_train,y_train)
print("the accuracy of classifying 20newsgroups using Naive Bayes:",mnb_count.score(x_count_test,y_test))
y_count_predict = mnb_count.predict(x_count_test)
from sklearn.metrics import  classification_report
print(classification_report(y_test,y_count_predict,target_names=news.target_names))

#tfidfVectorizer.
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()
x_tfidf_train = tfidf_vec.fit_transform(x_train)
x_tfidf_test = tfidf_vec.transform(x_test)
mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(x_tfidf_train,y_train)
print("the accuracy of classifying 20newsgroups with naive bayes:",mnb_tfidf.score(x_tfidf_test,y_test))
y_tfidf_predict = mnb_tfidf.predict(x_tfidf_test)
print(classification_report(y_test,y_tfidf_predict,target_names=news.target_names))
