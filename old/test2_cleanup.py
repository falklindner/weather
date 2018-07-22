import pandas as pd
import time 
import json
import urllib.request
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

with urllib.request.urlopen("https://alte-rs.ddnss.de/weather/processed/history.json") as url:
    raw_data = json.loads(url.read().decode())


df = pd.DataFrame(data=raw_data)



df['time'] = pd.to_datetime(df['time'])
df['temp'] = pd.to_numeric(df['temp'])
df['pressure'] = pd.to_numeric(df['pressure'])
df['humidity'] = pd.to_numeric(df['humidity'])
df = df.set_index('time')

# Rolling series with Window Width win_width 

win_width = '10min'
time_step = timedelta(minutes = 5)
rolling_avg = df.rolling(win_width,min_periods=5).mean()


# Remove all data, exept the rolling average for every time_step, beginning at init

init = rolling_avg.index[0]

for time in rolling_avg.index:
    if time-init > time_step:
        init = time
    else:
        rolling_avg = rolling_avg.drop(time)
  
        
# Label set is Weather Values (Rolling means) prediction_time later than now (time in multiples of win_width)

prediction_time = 20

# Features are all Weather Values (Rolling means), history time in the past from now

history_time = 20

# First label can only be predicted after history + prediction time

y = rolling_avg[ prediction_time + history_time : ].values

length = len(rolling_avg)

# Setting up feature matrix. Latest feature data can not predict anything, since there is no future for them in the set

x = np.zeros((length - (prediction_time + history_time) , 3 * history_time))

# Importing the weather values into the feature matrix x

for row in range(0 , length - (prediction_time + history_time) ):
    for data_col in range(0,history_time):
        for i in range(0,3):
            x[row,3*data_col+i]= rolling_avg.loc[rolling_avg.index[row + data_col]][i]



from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x)
x = imputer.transform(x)
imputer = imputer.fit(y)
y = imputer.transform(y)


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()


X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the Test set results
y_pred = regressor.predict(X_test)


from sklearn.metrics import mean_squared_error
y_true = [3, -0.5, 2, 7]

mean_squared_error(y_test, y_pred)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring= 'neg_mean_absolute_error', n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[101]:


title = "Learning Curves Our model"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

plot_learning_curve(regressor, title, X_train, y_train, cv=cv, n_jobs=4)
plt.show()


# Fazit: Das System lernt doch!



plt.scatter(X_train[:,0], y_train[:,0], color = 'red')
plt.scatter(X_train[:,0], regressor.predict(X_train)[:,0], color = 'blue')
plt.title('Temp Future vs. Past (Training set)')
plt.xlabel('Temperature X')
plt.ylabel('Temperature Y')
plt.show()


# In[103]:


# Visualising the Test set results
plt.scatter(X_test[:,0], y_test[:,0], color = 'red')
plt.scatter(X_test[:,0], regressor.predict(X_test)[:,0], color = 'blue')
plt.title('Temp Future vs. Past (Training set)')
plt.xlabel('Temperature X')
plt.ylabel('Temperature Y')
plt.show()

