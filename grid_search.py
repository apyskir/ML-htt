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
          "n_estimators": 50,
          "gamma": 1}

print "start splitting"
start = time.time()
trainSample, testSample = splitTrainIntoTT(train, 0.6)
end = time.time()
print "end splitting: ", end - start, " s"

params_grid = {}

grid = (
    {"min_child_weight": [2,4,6,8,10], "max_depth":[3,4,5,6,7]},
    {"gamma": [0, 0.2, 0.4, 0.6, 0.8, 1]},
    {"subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1], "colsample_bytree":[0.5, 0.6, 0.7, 0.8, 0.9, 1]},
    {"reg_alpha":[0.01, 0.1, 1, 10], "reg_lambda":[0.01, 0.1, 1, 10]}
)

#test purposes:
grid = (
    {"min_child_weight": [2,3], "max_depth":[5,6]},
    {"gamma": [0.08, 0.1]}
)

shrinkage = 3
geom_coeff = {1: [1,],
    2: [0.5, 2],
    3: [0.5, 1, 2],
    4: [0.2, 0.5, 2, 5],
    5: [0.2, 0.5, 1, 2, 5]
}

nFolds = 4
cv_generator = sklearn.model_selection.StratifiedKFold(m_splits=nFolds, shuffle=True)

print "start GridSearchCV"
start = time.time()

for x in grid:
    params_grid = x
    cv_result = xgb.cv(params, xgb.DMatrix(trainSample[features], trainSample["signal"]), num_boost_round=100, nfold=nFolds, metrics="logloss", early_stopping_rounds=10)
    params["n_estimators"] = cv_result.shape[0]
    print cv_result.shape[0]
    grid = GridSearchCV(estimator=XGBClassifier(**params), param_grid = params_grid, cv = cv_generator, iid = False, scoring = "precision", verbose=2, n_jobs=4)
    grid.fit(trainSample[features], trainSample["signal"])
    params.update(grid.best_params_)
    for key in params_grid.keys():
        tmp = params_grid[key]
        if tmp[1]-tmp[0] == tmp[len(tmp)-1] - tmp[len(tmp)-2]:
            mid = grid.best_params_[key]
            diff = float(tmp[1]-tmp[0])
            params_grid[key] = getNewArythmList(mid, diff, shrinkage, key)
        else:
            mid = grid.best_params_[key]
            params_grid[key] = [mid * scale for scale in geom_coeff[2*shrinkage-1]]
    grid = GridSearchCV(estimator=XGBClassifier(**params), param_grid = params_grid, cv = cv_generator, iid = False, scoring = "precision", verbose=2, n_jobs=4)
    grid.fit(trainSample[features], trainSample["signal"])
    params.update(grid.best_params_)

print params

end = time.time()
print "end GridSearchCV: ", end - start, " s"
