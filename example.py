import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import metrics
from configs import TI_FACTOR_CONFIG_SHORT_TERM
from indicator import make_ti_factor
import dataclient as dt
from constants import  CLOSE
import matplotlib.pyplot as plt


def feature_importance(cl_rf, ti_factor_fileds, plot= False):
    features = np.array(ti_factor_fileds)
    importances = cl_rf.feature_importances_
    indices = np.argsort(importances)
    if plot:
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), features[indices])
        plt.xlabel('Relative Importance')
        plt.subplots_adjust(left=0.35)
        plt.show()
    return features[indices], importances[indices]


if __name__ == '__main__':
    #load data
    data_df = dt.load_data()
    data_df =  dt.cleanData(data_df, 'price' ,checklist=('nan', 'nonpositive', 'outlier'), n_std=3)
    data_df =  dt.cleanData(data_df, 'quantity', checklist=('nan', 'nonpositive'))
    # plt.plot(data_df[TICK_PRICE_FIELD])
    # plt.show()
    # aggregate data
    resampled_data_df = dt.bar_from_tick(data_df, freq='1min', non_trading_hours=[('04:15','04:30'),('05:00','06:00')])

    #make technical indictor
    ti_factor_fileds, resampled_data_df = make_ti_factor(resampled_data_df, TI_FACTOR_CONFIG_SHORT_TERM)
    #Only keep signal and hist field of macd
    ti_factor_fileds.remove('MACD13_21_8')

    #make label
    label = 'Trend'
    resampled_data_df[label] = resampled_data_df[CLOSE].pct_change(5).shift(-5).apply(lambda x: 1 if x>0 else 0)

    resampled_data_df = resampled_data_df.dropna()

    X = resampled_data_df[ti_factor_fileds]  # Features
    y = resampled_data_df[label]  # Labels

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    param_grid = {
        'bootstrap': [True],
        'min_samples_leaf': [0.1, 0.05],
        'n_estimators': [30, 60, 100]
    }
    rf = RandomForestClassifier(oob_score=True, random_state=1)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=0, n_jobs=6)
    grid_search.fit(X_train, y_train)
    cl_rf = grid_search.best_estimator_
    pre = cl_rf.predict(X_test)

    print(metrics.classification_report(y_test, pre))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, pre))

    # plot roc curve
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, pre)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    #check feature importance
    feature_importance(cl_rf, ti_factor_fileds, True)
