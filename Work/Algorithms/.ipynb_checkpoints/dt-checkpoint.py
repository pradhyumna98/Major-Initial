from sklearn.tree import DecisionTreeClassifier
def get_dt_model(X_train, X_test, y_train, y_test):
    model_dt = DecisionTreeClassifier()

    model_dt.fit(X_train , y_train)

    return model_dt , model_dt.score(X_test , y_test)