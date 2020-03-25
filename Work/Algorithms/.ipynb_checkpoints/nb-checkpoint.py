from sklearn.naive_bayes import *

def get_nb_model(X_train, X_test, y_train, y_test):
    model_nb = BernoulliNB()
    model_nb.fit(X_train,y_train)
    
    return model_nb , model_nb.score(X_test , y_test)