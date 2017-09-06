print "start import"
from xgbbase import *
print "end import"

#print(signal.head(10))
features = ["pt_1", "pt_2", "extramuon_veto"]

params = {"objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          #"silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1,
          "verbose": 2,
          "print_every_n": 10}

num_trees=250

print "start splitting"
start = time.time()
a = splitTrainIntoTTV(train, 0.6, 0.2)
end = time.time()
print "end splitting: ", end - start, " s"
trainSample = a[0]
testSample = a[1]
valSample = a[2]

model = xgb.sklearn.XGBClassifier(objective = "binary:logistic",
          learning_rate = 0.3,
          max_depth = 5,
          min_child_weight = 3,
          #silent = 1,
          subsample = 0.7,
          colsample_bytree = 0.7,
          seed = 1,
          #verbose = 2,
          #print_every_n = 10)

print "start train"
start = time.time()
model.fit(trainSample[features], trainSample[signal])
end = time.time()
print "end train: ", end - start, " s"

print "start predict test"
start = time.time()
results = model.predict(xgb.DMatrix(testSample[features]))
end = time.time()
print "end predict: ", end - start, " s"

print "start predict val"
start = time.time()
results = model.predict(xgb.DMatrix(valSample[features]))
end = time.time()
print "end predict: ", end - start, " s"
print results
