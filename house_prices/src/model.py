from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import json
import os
import lightgbm as lgb
import pdb

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   

def rmsle_cv(model, x_train, y_train):
    n_folds = 5
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train.values)
    rmse= np.sqrt(-cross_val_score(model, x_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

class Model:
    def __init__(self, model_name) -> None:
        self.config = json.load(open(self.get_config_path(), "r"))
        self.model = self.get_model(model_name)

    def get_config_path(self):
        src_dir = os.path.dirname(__file__)
        root = os.path.dirname(src_dir)
        config_dir = os.path.join(root, "config")
        return os.path.join(config_dir, "config.json")

    def build(self, x_train, y_train):
        pipe = self.lasso()
        pipe.fit(x_train, y_train)
        self.model = pipe
        print("Build Success")

    def get_model(self, model_name):
        if model_name == "lasso":
            return self.lasso()
        elif model_name == "enet":
            return self.enet()
        elif model_name == "krr":
            return self.krr()
        elif model_name == "gboost":
            return self.gboost()
        elif model_name == "lgb":
            return self.lgb_model()
        elif model_name == "avg":
            return self.avg_model()
        else:
            raise Exception("wrong model name: {}".format(model_name))

    def lasso(self):
        return make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
    
    def enet(self):
        return make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

    def krr(self):
        return KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    
    def gboost(self):
        return GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10, 
                                loss='huber', random_state =5)

    def lgb_model(self):
       return lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

    def avg_model(self):
        model_names = self.config["model"]["average_model"]["model_list"]
        model_list = [self.get_model(name) for name in model_names]
        return AveragingModels(model_list)
    
    
    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        y_pred = np.exp(y_pred)
        return y_pred