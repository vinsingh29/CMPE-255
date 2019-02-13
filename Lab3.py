
# coding: utf-8

# In[39]:


def load_data():
    data = []
    data_labels = []
    with open("./pos_tweets.txt") as f:
        for i in f: 
            data.append(i) 
            data_labels.append('pos')

    with open("./neg_tweets.txt") as f:
        for i in f: 
            data.append(i)
            data_labels.append('neg')
    #print(data)
    return data, data_labels



def transform_to_features(data):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(
        analyzer = 'word',
        lowercase = False,
    )
    features = vectorizer.fit_transform(
        data
    )
    features_nd = features.toarray()
    return features_nd

def train_then_build_model(data_labels, features_nd,data):
    from sklearn.cross_validation import train_test_split
    # TODO : set training % to 80%.
    X_train, X_test, y_train, y_test  = train_test_split(
        features_nd, 
        data_labels,
        train_size=0.80, 
        random_state=1234)

    from sklearn.linear_model import LogisticRegression
    log_model = LogisticRegression()
    
    
    log_model = log_model.fit(X=X_train, y=y_train)
    y_pred = log_model.predict(X_test)
    print("Prediction for 10th tweet:",y_pred[9])
    
  
    
    # print first 10th prediction in this format:
    # ::{prediction}::{tweet}
    # TODO
    data_val=int((len(data)*0.8))
    
    for i in range(10):
        print("Predication is %s, Tweet : %s" % (y_pred[i],data[data_val+i] ))
   
    
    
    # print accuracy
    from sklearn.metrics import accuracy_score
    # TODO
    accuracy =(accuracy_score(y_pred, y_test))
    print("Accuracy={}".format(accuracy*100))
    
def process():
    data, data_labels = load_data()
    features_nd = transform_to_features(data)
    train_then_build_model(data_labels, features_nd,data)


process()

