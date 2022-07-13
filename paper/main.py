# Import libraries
import pandas as pd # for data analsysis
import numpy as np # to handle data in a vectorized manner
import seaborn as sns # for visualization
from sklearn.model_selection import RandomizedSearchCV # for hyperparameters tuning
from sklearn.model_selection import cross_val_score # for cross-validation evaluation
from sklearn.metrics import mean_squared_error # to calculate the RMSE


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

import matplotlib.pyplot as plt

import time
import pickle




def prep(df_dummy, n_step = 1):
    X = df_dummy.drop(columns={'ENGINE_1_FUEL_CONSUMPTION','ENGINE_2_FUEL_CONSUMPTION'})
    X = X.drop(columns={'ENGINE_2_FLOWRATEA','ENGINE_1_FLOWRATE','ENGINE_2_FLOWRATE','ENGINE_1_FLOWRATEA','Dati'})
    # feature engineering
    df_dummy = df_dummy.assign(SOGmSTW=lambda x: x.SOG -  x.STW)
    df_dummy['pLONGITUDE'] = (df_dummy['LONGITUDE'].shift(periods=-1)).fillna(-123.2733)
    df_dummy['pLATITUDE'] = df_dummy['LATITUDE'].shift(periods=-1).fillna(49.3859 )
    df_dummy = df_dummy.assign(DISP=lambda x: ((x.LONGITUDE -  x.pLONGITUDE)**2 + (x.LATITUDE-x.pLATITUDE)**2)**(0.5))
    df_dummy = df_dummy.assign(PITCH=lambda x: (x.PITCH_1 +  x.PITCH_2)/2)
    df_dummy = df_dummy.assign(SPEED=lambda x: (x.SPEED_1 +  x.SPEED_2)/2)
    df_dummy = df_dummy.assign(THRUST=lambda x: (x.THRUST_1 +  x.THRUST_2)/2)
    df_dummy = df_dummy.assign(TORQUE=lambda x: (x.TORQUE_1 +  x.TORQUE_2)/2)
    df_dummy = (df_dummy.assign(OUT=lambda x: (x.ENGINE_1_FUEL_CONSUMPTION +  x.ENGINE_2_FUEL_CONSUMPTION/2/x.DISP*60/10**6)))

    if n_step>1:
        df_dummy['OUT'] = df_dummy['OUT'].shift(-n_step)
        df_dummy.dropna(inplace=True)

    #feature selection
    y = df_dummy['OUT'].clip(upper=1000)
    X = df_dummy[['PITCH','SPEED','STW','WIND_SPEED','WIND_ANGLE','SOG','SOGmSTW','HEADING','DISP','TORQUE']]



    # # For using later in Neural networks
    # DF = X.copy()
    # DF = DF.join(df_dummy['OUT'].clip(upper=1000))
    # DF.to_csv('./data/in_out_NN.csv')

    return X, y


def plot(y_test, y_pred, fig_name):
    plt.figure()
    ax1 = sns.distplot(y_test, color='r', hist=False, label='actual')
    ax2 = sns.distplot(y_pred,  color='b', hist=False, label='prediction', ax=ax1)
    plt.legend(fontsize=5)
    plt.savefig(fig_name)


def lrm(X_train_scaled, y_train, type):
    if type == 'lr':
        lin_reg = LinearRegression()
    elif type == 'lasso':
        lin_reg = Lasso(alpha=0.1)
    elif type == 'ridge':
        lin_reg = Ridge(alpha=10)
    else:
        raise Exception("The type should be either of these choices ['lr', 'lasso','ridge']")


    # Model evaluation by cross-validation
    lin_reg_score = cross_val_score(lin_reg, X_train_scaled, y_train, verbose = 2)

    lin_reg.fit(X_train_scaled, y_train)

    return lin_reg_score, lin_reg.coef_, lin_reg






def main():
    dict = {} # feature importance for different methods
    # Save the data frame to a file
    df_dummy = pd.read_pickle('./data/df_dummy.pkl')
    X, y = prep(df_dummy, n_step=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    # Scaler for X
    scaler_x = MinMaxScaler(feature_range=(0,1))
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    ## linear regression
    st = time.time()
    mean_score_train, fi_lr, model = lrm(X_train_scaled, y_train, type = 'lr')  # type =['lr', 'lasso','ridge']
    et = time.time()
    print(f'R^2 Validation: {mean_score_train.mean()}')
    # saving the results
    dict['lr'] = {}
    dict['lr']['importance'] = fi_lr
    dict['lr']['train_time'] = et - st
    #test
    st = time.time()
    y_pred = model.predict(X_test_scaled)
    et = time.time()
    print(f'R^2 Test: {model.score(X_test_scaled, y_test)}')
    print(f'RMSE Test: {np.sqrt(mean_squared_error(y_test, y_pred))}')
    dict['lr']['test_time'] = et - st
    dict['lr']['R2'] = [mean_score_train.mean(), model.score(X_test_scaled, y_test), np.sqrt(mean_squared_error(y_test, y_pred))]
    dict['lr']['model'] = model
    # plot the result
    plot(y_test, y_pred, 'lin_reg.eps')

    print("done...")

    ## Poly regression
    poly_features = PolynomialFeatures(degree = 2)
    # Returns a transformed version of X with new combinations of features
    X_train_scaled_poly = poly_features.fit_transform(X_train_scaled)
    X_test_scaled_poly = poly_features.fit_transform(X_test_scaled)
    st = time.time()
    mean_score_train, fi_lr, model = lrm(X_train_scaled_poly, y_train, type = 'ridge')  # type =['lr', 'lasso','ridge']
    et = time.time()
    # print(f'R^2 Validation: {mean_score_train.mean()}')
    # saving the results
    dict['pr'] = {}
    dict['pr']['importance'] = fi_lr[1:11]
    dict['pr']['train_time'] = et - st
    # test
    st = time.time()
    y_pred = model.predict(X_test_scaled_poly)
    et = time.time()
    # print(f'R^2 Test: {model.score(X_test_scaled_poly, y_test)}')
    # print(f'RMSE Test: {np.sqrt(mean_squared_error(y_test, y_pred))}')
    dict['pr']['test_time'] = et - st
    dict['pr']['R2'] = [mean_score_train.mean(), model.score(X_test_scaled_poly, y_test), np.sqrt(mean_squared_error(y_test, y_pred))]
    dict['pr']['model'] = model
    # plot the result
    plot(y_test, y_pred, 'pol_reg.eps')

    ## Desicion tree
    # Hyperparameter values to feed to the RandomizedSearchCV
    param_grid = {'max_features': ['auto', 'sqrt'], # Number of features to consider at every split
              'max_depth': np.arange(5, 41, 5), # Maximum number of levels in tree #41
              'min_samples_split': [5, 10, 20, 40], # Minimum number of samples required to split a node #40
              'min_samples_leaf': [2, 6, 12, 24], # Minimum number of samples required at each leaf node #24
              }

    # Instantiate a RandomizedSearchCV on a DecisionTreeRegressor model with 100 iterations
    st = time.time()
    tree_reg = RandomizedSearchCV(estimator = DecisionTreeRegressor(), param_distributions = param_grid, n_iter = 10, verbose = 2, n_jobs = -1) #10
    tree_reg.fit(X_train, y_train)
    et = time.time()
    # saving the results
    dict['dt'] = {}
    dict['dt']['train_time'] = et - st
    # print(f'R^2 Validation: {tree_reg.best_score_}')
    dict['dt']['importance'] = tree_reg.best_estimator_.feature_importances_
    # test
    st = time.time()
    y_pred = tree_reg.predict(X_test)  # we dont use the transformed version
    et = time.time()
    # print(f'R^2 Test: {tree_reg.score(X_test, y_test)}')
    # print(f'RMSE Test: {np.sqrt(mean_squared_error(y_test, y_pred))}')
    dict['dt']['test_time'] = et - st
    dict['dt']['R2'] = [tree_reg.best_score_, tree_reg.score(X_test, y_test), np.sqrt(mean_squared_error(y_test, y_pred))]
    dict['dt']['model'] = tree_reg.best_estimator_
    # plot the result
    plot(y_test, y_pred, 'tree_reg.eps')


    ## Random forest
    # Hyperparameter values to feed to the RandomizedSearchCV same as DT
    st = time.time()
    rfor_reg = RandomizedSearchCV(RandomForestRegressor(), param_distributions = param_grid, n_iter = 10, verbose = 2, n_jobs = -1)
    rfor_reg.fit(X_train, y_train)
    et = time.time()
    # saving the results
    dict['rf'] = {}
    dict['rf']['train_time'] = et - st
    # print(f'R^2 Validation: {rfor_reg.best_score_}')
    dict['rf']['importance'] = rfor_reg.best_estimator_.feature_importances_
    # test
    st = time.time()
    y_pred = rfor_reg.predict(X_test)
    et = time.time()
    # print(f'R^2 Test: {rfor_reg.score(X_test, y_test)}')
    # print(f'RMSE Test: {np.sqrt(mean_squared_error(y_test, y_pred))}')
    dict['rf']['test_time'] = et - st
    dict['rf']['R2'] = [rfor_reg.best_score_, rfor_reg.score(X_test, y_test), np.sqrt(mean_squared_error(y_test, y_pred))]
    dict['rf']['model'] = rfor_reg.best_estimator_
    # plot the result
    plot(y_test, y_pred, 'rf_reg.eps')



    ## Adaboost
    # Hyperparameter values to feed to the RandomizedSearchCV
    param_grid = {"learning_rate"   : [0.01, 0.1, 0.3], #,0.3
              "loss"            : ['linear', 'square', 'exponential'] #  ,'exponential'
              }
    st = time.time()
    ada_reg = RandomizedSearchCV(AdaBoostRegressor(DecisionTreeRegressor(min_samples_split = 40, min_samples_leaf = 2, max_depth = 40),
                                                   n_estimators=100), param_distributions = param_grid, n_iter = 10, verbose = 2, n_jobs = -1)
    ada_reg.fit(X_train, y_train)
    et = time.time()
    # saving the results
    dict['ab'] = {}
    dict['ab']['train_time'] = et - st
    # print(f'R^2 Validation: {ada_reg.best_score_}')
    dict['ab']['importance'] = ada_reg.best_estimator_.feature_importances_
    # test
    st = time.time()
    y_pred = ada_reg.predict(X_test)
    et = time.time()
    # print(f'R^2 Test: {ada_reg.score(X_test, y_test)}')
    # print(f'RMSE Test: {np.sqrt(mean_squared_error(y_test, y_pred))}')
    dict['ab']['test_time'] = et - st
    dict['ab']['R2'] = [ada_reg.best_score_, ada_reg.score(X_test, y_test), np.sqrt(mean_squared_error(y_test, y_pred))]
    dict['ab']['model'] = ada_reg.best_estimator_
    # plot the result
    plot(y_test, y_pred, 'ab_reg.eps')


    ## Gradient boosting
    # Hyperparameter values to feed to the RandomizedSearchCV
    param_grid = {"learning_rate"   : [0.01, 0.1, 0.3],
              "subsample"        : [0.5, 1.0],
              'max_depth'        : np.arange(5, 41, 5), # 40
              "max_features"     : ['auto', 'sqrt'],
              "min_samples_split": [5, 10, 20,40], #, 20,40
              "min_samples_leaf" : [2, 6, 12, 24] #  ,12, 24
              }
    st = time.time()
    grad_reg = RandomizedSearchCV(GradientBoostingRegressor(), param_distributions = param_grid, n_iter = 10, verbose = 2, n_jobs = -1)

    grad_reg.fit(X_train, y_train)
    et = time.time()
    # print(f'R^2 Validation: {grad_reg.best_score_}')
    # saving the results
    dict['gb'] = {}
    dict['gb']['train_time'] = et - st
    # print(f'R^2 Validation: {ada_reg.best_score_}')
    dict['gb']['importance'] = grad_reg.best_estimator_.feature_importances_
    # test
    st = time.time()
    y_pred = grad_reg.predict(X_test)
    et = time.time()
    # print(f'R^2 Test: {grad_reg.score(X_test, y_test)}')
    # print(f'RMSE Test: {np.sqrt(mean_squared_error(y_test, y_pred))}')
    dict['gb']['test_time'] = et - st
    dict['gb']['R2'] = [grad_reg.best_score_, grad_reg.score(X_test, y_test), np.sqrt(mean_squared_error(y_test, y_pred))]
    dict['gb']['model'] = grad_reg.best_estimator_
    # plot the result
    plot(y_test, y_pred, 'gb_reg.eps')


    ## XGBoost
    param_grid = {"learning_rate"   : [0.01, 0.1, 0.3] ,
                  'max_depth'        : np.arange(5, 41, 5), #40
                  "min_child_weight" : [1, 3, 5, 7],
                  "gamma"            : [0.0, 0.1, 0.2, 0.3, 0.4],
                  "colsample_bytree" : [0.3, 0.4, 0.5, 0.7]
                  }
    st = time.time()
    xgb_reg = RandomizedSearchCV(XGBRegressor(), param_distributions = param_grid, n_iter = 10, verbose = 2, n_jobs = -1) #10
    xgb_reg.fit(X_train, y_train)
    et = time.time()
    print(f'R^2 Validation: {xgb_reg.best_score_}')
    # saving the results
    dict['xgb'] = {}
    dict['xgb']['train_time'] = et - st
    # print(f'R^2 Validation: {xgb_reg.best_score_}')
    # print([i for i in xgb_reg.best_estimator_.get_booster().get_score(importance_type='weight').keys()])
    dict['xgb']['importance'] = [i for i in xgb_reg.best_estimator_.get_booster().get_score(importance_type='weight').values()]
    # test
    st = time.time()
    y_pred = xgb_reg.predict(X_test)
    et = time.time()
    # print(f'R^2 Test: {xgb_reg.score(X_test, y_test)}')
    # print(f'RMSE Test: {np.sqrt(mean_squared_error(y_test, y_pred))}')
    dict['xgb']['test_time'] = et - st
    dict['xgb']['R2'] = [xgb_reg.best_score_, xgb_reg.score(X_test, y_test), np.sqrt(mean_squared_error(y_test, y_pred))]
    dict['xgb']['model'] = xgb_reg.best_estimator_
    # plot the result
    plot(y_test, y_pred, 'xgb_reg.eps')




    with open('saved_dictionary.pkl', 'wb') as f:
        pickle.dump([dict], f)


if __name__ == '__main__':
    main()
