import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import svm
from random import choice
from data import question_bank, replies

def train_data(tv, model, file):
    # Read from csv.
    possible_ans = pd.read_csv(file)

    # Split data into train and test
    ans = possible_ans['Ans']
    label = possible_ans["Label"]
    ans_train, ans_test,label_train, label_test = train_test_split(ans,label,test_size = 0.1, shuffle = True)

    # Train data
    features = tv.fit_transform(ans_train)
    model.fit(features,label_train)


def check_ans(tv, model, user_ans):
    features_test = tv.transform(pd.Series(user_ans))
    return model.predict(features_test)[-1]


tv = TfidfVectorizer()
model = svm.SVC()

# Initializing
score = 0
count = 0

for question_number in question_bank:
    print(f"\nBot: {question_number['question']}")
    train_data(tv, model, question_number['file'])
    user_ans = input("> ")
    result = check_ans(tv, model, user_ans)
    # print(result)

    count += 1
    
    if result == "Right":
        print(f"Bot: {choice(replies)}")
        score+=10
    else:
        print(f"Bot :Hmm..., I'll give you a hint - {question_number['hint']}")
        user_ans = input("> ")
        result = check_ans(tv, model, user_ans)
        if result == "Right":
            score += 5
        print(f"Bot: {choice(replies)}")

final_score = (score/(count*10)) * 100
print(f"\nScore: {final_score}\n")