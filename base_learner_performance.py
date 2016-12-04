from sklearn.svm import SVC
from reader import read_train_test

X_train, y_train = read_train_test("DIFFICULT_TRAIN.csv")
X_test, y_test = read_train_test("DIFFICULT_TEST.csv")

print(sum(y_train == 0))
print(sum(y_train == 1))
print(sum(y_train == 2))
print(y_train.shape)
print(sum(y_test == 0))
print(sum(y_test == 1))
print(sum(y_test == 2))
print(y_test.shape)
svc = SVC()
svc.fit(X_train, y_train)
print(svc.score(X_test, y_test))
