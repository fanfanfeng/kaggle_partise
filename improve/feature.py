
measurements = [{'city':"Dubai",'temperature':33.},
                {'city':"London","temperature":12.},
                {'city':"san fransisco",'temperature':18.}]
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
print(vec.fit_transform(measurements).toarray())
print(vec.get_feature_names())