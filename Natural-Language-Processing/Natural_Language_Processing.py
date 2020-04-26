# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset 
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) # quoting=3 ignores double quotes

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# Corpus = collection of texts
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i])
    review = review.lower()
    # to compare the stop words with words in the review, which is a string, we need
    # to make the string into list of words
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

'''#---------------Naive Bayes---------------------------#'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Naive Bayes Classification: \nAccuracy =',(cm[0][0]+cm[1][1])/200,
      '\nPecision =',cm[0][0]/(cm[0][0]+cm[1][0]),
      '\nRecall  =',cm[0][0]/(cm[0][0]+cm[0][1]),
      '\nF1 Score =',2*(cm[0][0]/(cm[0][0]+cm[1][0]))*(cm[0][0]/(cm[0][0]+cm[0][1]))/((cm[0][0]/(cm[0][0]+cm[1][0]))+(cm[0][0]/(cm[0][0]+cm[0][1]))))

'''---------------- Decision Tree Classification --------------- '''
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train) 

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('\nDecision Tree Classification: \nAccuracy =',(cm[0][0]+cm[1][1])/200,
      '\nPecision =',cm[0][0]/(cm[0][0]+cm[1][0]),
      '\nRecall  =',cm[0][0]/(cm[0][0]+cm[0][1]),
      '\nF1 Score =',2*(cm[0][0]/(cm[0][0]+cm[1][0]))*(cm[0][0]/(cm[0][0]+cm[0][1]))/((cm[0][0]/(cm[0][0]+cm[1][0]))+(cm[0][0]/(cm[0][0]+cm[0][1]))))

'''-------------- Random Forest Classification ---------------- '''
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('\nRandom Forest Classification: \nAccuracy =',(cm[0][0]+cm[1][1])/200,
      '\nPecision =',cm[0][0]/(cm[0][0]+cm[1][0]),
      '\nRecall  =',cm[0][0]/(cm[0][0]+cm[0][1]),
      '\nF1 Score =',2*(cm[0][0]/(cm[0][0]+cm[1][0]))*(cm[0][0]/(cm[0][0]+cm[0][1]))/((cm[0][0]/(cm[0][0]+cm[1][0]))+(cm[0][0]/(cm[0][0]+cm[0][1]))))