```python 

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

def get_score(n_estimators):
    
    my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators, random_state=0))
    ])
    
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                 cv=3,
                                 scoring='neg_mean_absolute_error')
    return scores.mean()
	
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(list(results.keys()), list(results.values()))
plt.show()


```
