import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.feature_selection import f_classif, chi2, SelectKBest
from reader import read_train_test
import matplotlib.pyplot as plt

X_train, y_train = read_train_test('MODERATE_TRAIN.csv')
X_test, y_test = read_train_test("MODERATE_TEST.csv")
scores = chi2(X_train, y_train)[0]
# print(scores)
# print(np.sort(scores[-100:]))
selector = SelectKBest(chi2, k=25)
selector.fit(X_train, y_train)
X_test = selector.transform(X_test)
true_support = selector.get_support().astype(np.int)
print(np.where(true_support == True)[0])
svc = SVC()
svc.fit(selector.transform(X_train), y_train)
svm_res = svc.predict(X_test)

print(svc.score(X_test, y_test))
print(f1_score(y_test, svm_res))
#
# plt.plot(np.sort(scores))
# plt.show()
#
diffs = []
for i in range(50, 4050, 50):
    fake_X_train = X_train[-i:, :]
    fake_y_train = y_train[-i:]
    print(fake_X_train.shape)
    print(fake_y_train.shape)
    fake_selector = SelectKBest(chi2, k=25)
    fake_selector.fit(fake_X_train, fake_y_train)
    fake_support = fake_selector.get_support().astype(np.int)
    diff = np.sum((np.subtract(true_support, fake_support) > 0).astype(np.int))
    print(diff)
    diffs.append(diff)

plt.plot(diffs)
plt.show()


