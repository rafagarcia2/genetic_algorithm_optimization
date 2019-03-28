from sklearn.neighbors import KNeighborsClassifier  



def knn(n_neighbors):
    classifier = KNeighborsClassifier(n_neighbors=5)  
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    return np.mean(y_pred != y_test)