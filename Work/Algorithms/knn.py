from sklearn.neighbors import KNeighborsClassifier

def get_knn_model(X_train, X_test, y_train, y_test):
    model_knn = KNeighborsClassifier()
    model_knn.fit(X_train,y_train)
    
    return model_knn , model_knn.score(X_test , y_test)