from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


class Model:
    def __init__(self, model_name) -> None:
        self.model = {
            "lasso": self.lasso(),
            "enet": self.enet(),
            "krr": self.krr(),
            "gboost": self.gboost()
        }[model_name]


    def build(self, x_train, y_train):
        pipe = self.lasso()
        pipe.fit(x_train, y_train)
        self.model = pipe
        print("Build Success")

    def rmsle_cv(self, x_train, y_train):
        n_folds = 5
        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train.values)
        rmse= np.sqrt(-cross_val_score(self.model, x_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
        return(rmse)

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

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred