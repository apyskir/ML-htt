print "start import"
from xgbbase import *
print "end import"
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics

#print(signal.head(10))
features = ["pt_1", "pt_2", "extramuon_veto"]

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

print "start splitting"
start = time.time()
trainSample, testSample = splitTrainIntoTT(train, 0.6)
end = time.time()
print "end splitting: ", end - start, " s"

params_grid = {}

grid = (
    {"min_child_weight": [2,4,6,8,10], "max_depth":[2,3,4,5,6]},
    {"gamma": [0, 0.1, 0.2, 0.5, 1]},
    {"subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 10], "colsample_bytree":[0.5, 0.6, 0.7, 0.8, 0.9, 10]},
    {"reg_alpha":[0.01, 0.1, 1, 10], "reg_lambda":[0.01, 0.1, 1, 10]}
)

#test purposes:
grid = (
    {"min_child_weight": [2,4]},
    {"gamma": [0, 0.1]}
)

cv_result = xgb.cv()

print "start GridSearchCV"
start = time.time()

for x in grid:
    params_grid = x
    grid = GridSearchCV(estimator=XGBClassifier(**params), param_grid = params_grid, cv = 3, scoring = "precision", verbose=2)
    grid.fit(trainSample[features], trainSample["signal"])
    print grid.get_params()

end = time.time()
print "end GridSearchCV: ", end - start, " s"
