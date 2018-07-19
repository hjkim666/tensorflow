from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
cancer = load_breast_cancer()

X = cancer.data

X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    random_state=0)

svm = SVC(C=100)
svm.fit(X_train, y_train)
print("test accuracy: {:.2f}".format(svm.score(X_test, y_test)))

### scale
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled,
                                                    cancer.target,
                                                    random_state=0)


svm.fit(X_train_scaled, y_train)
print("scaled test accuracy: {:.2f}".format(svm.score(X_test_scaled
                                                      , y_test)))

