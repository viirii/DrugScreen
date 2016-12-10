from mclearn.active import ActiveLearner, run_active_learning_expt
from mclearn.heuristics import random_h, qbb_kl_h
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from reader import read_train_test

# TODO: under construction!


X_train, y_train = read_train_test("EASY_TRAIN.csv")
X_test, y_test = read_train_test("EASY_TEST.csv")

classifier = SVC(kernel='rbf', gamma=0.1, C=10, cache_size=2000, class_weight='auto', probability=True)
committee = BaggingClassifier(classifier, n_estimators=11, n_jobs=-1, max_samples=300)
heuristic = qbb_kl_h
initial_n = 50
training_size = 2500
sample_size = 300
verbose = True
committee_samples = 300
pool_n = 300
C = 1
active_learner = ActiveLearner(classifier=classifier,
                               heuristic=heuristic,
                               initial_n=initial_n,
                               training_size=training_size,
                               sample_size=sample_size,
                               verbose=verbose,
                               committee=committee,
                               committee_samples=committee_samples,
                               pool_n=pool_n,
                               C=C,
                               random_state=1,
                               pool_random_state=1)
active_learner.fit(X_train, y_train, X_test, y_test)
print(active_learner.learning_curve_)
print(active_learner.candidate_selections)
print(active_learner.accuracy_fn)
