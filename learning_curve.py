from xgbbase import *
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from xgboost.sklearn import XGBClassifier

def draw_learning_curve(model, X, y, train_sizes, cv=5, modelName="your_favorite_model", verbosity=0):
    print "start learning curve calculations"
    start = time.time()
    train_sizes, train_scores, valid_scores = learning_curve(model, X, y, train_sizes = train_sizes, cv = cv, scoring = "precision", verbose = verbosity, n_jobs=4)
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)
    valid_scores_mean = np.mean(valid_scores, axis = 1)
    valid_scores_std = np.std(valid_scores, axis = 1)

    fig = plt.figure(figsize=(10,6), dpi=100)

    plt.plot(train_sizes, train_scores_mean, label="Train score", color = "r")
    plt.plot(train_sizes, valid_scores_mean, label="Cross-validation score", color = "g")
    plt.fill_between(train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2, color = "r")
    plt.fill_between(train_sizes,
        valid_scores_mean - valid_scores_std,
        valid_scores_mean + valid_scores_std,
        alpha=0.2, color = "g")

    plt.legend(loc="best")
    plt.title("Learning curve")
    plt.xlabel("Training sample size")
    plt.ylabel("Precision")
    #plt.show()
    plt.savefig("fig/" + modelName + "_learning_curve.png")
    end = time.time()
    print "end learning curve calculations: ", end - start, " s"

features = ["pt_1", "pt_2","eta_1", "eta_2", "extramuon_veto"]
for x in train.columns:
    print x

params = {"objective": "binary:logistic",
          "learning_rate": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "random_state": 1,
          "print_every_n": 10,
          "eval_metric": "logloss",
          "n_estimators": 50}

#X, y = make_classification(n_samples = 1000, n_features = 10, n_informative = 9, n_redundant = 0)
#draw_learning_curve(XGBClassifier(**params), X, y, [0.1, 0.3, 0.5, 0.7, 0.9], 10, "XGB")

draw_learning_curve(XGBClassifier(**params), train[features], train["signal"], [0.01, 0.05, 0.1, 0.3, 0.5, 0.9], modelName="XGB", verbosity = 1)
