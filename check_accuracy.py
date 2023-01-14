import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import svm

tv = TfidfVectorizer()
model = svm.SVC()

# Read csv
possible_ans = pd.read_csv('q1_copy.csv')
total_rows = len(possible_ans)
ratio = 1 - (total_rows-1)/total_rows

# Split data into train and test
ans = possible_ans['Ans']
label = possible_ans["Label"]
ans_train, ans_test,label_train, label_test = train_test_split(ans,label,test_size = ratio, shuffle = True)

# Train data
features = tv.fit_transform(ans_train)
model.fit(features,label_train)

# Check accuracy with test data
features_test = tv.transform(ans_test)
# features_test = tv.transform(pd.Series("I don't know"))
print("Accuracy: {}".format(model.score(features_test,label_test)))
print("Prediction: ", model.predict(features_test))
print("Actual score: ",label_test.unique())