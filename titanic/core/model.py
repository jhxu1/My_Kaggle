from core.data_process import process
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.metrics import accuracy_score
from sklearn import set_config

class Model:
    def __init__(self) -> None:
        self.model = None


    def build(self, x_train, y_train):
    #handling missing values
        tf1 = ColumnTransformer(transformers=[
            ("AgeImputer", SimpleImputer(), [2]),
            ("EmbarkedImputer", SimpleImputer(strategy="most_frequent"), [-1])
        ], remainder="passthrough")

        #encoding categorical features
        tf2 = ColumnTransformer(transformers=[
            ("SexEncoder", OrdinalEncoder(), [3]),
            ("EmbarkedOneHot", OneHotEncoder(sparse=False, handle_unknown="ignore"), [1,7])
        ], remainder="passthrough")

        # Scaling
        tf3 = ColumnTransformer([
            ('scale',StandardScaler(),slice(0,-1))
        ])

        # Model
        tf4 = LogisticRegression()

        pipe = make_pipeline(tf1,tf2,tf3,tf4)

        # Display Pipeline

        set_config(display='diagram')
        pipe.fit(x_train, y_train)
        self.model = pipe
        print("Build Success")

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred

def evaluate(y_pred, y_test):
    accuracy = round(accuracy_score(y_test, y_pred),3)*100
    print(f"The accuracy of the model is: {accuracy}%")
     