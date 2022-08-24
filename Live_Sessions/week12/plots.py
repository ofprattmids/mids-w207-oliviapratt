# standard
import pandas as pd
import numpy as np
import os
import plots # my own functions

# analysis
import patsy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Masking
tf.keras.backend.set_floatx('float64')

# plots
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from PIL import Image

# metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, matthews_corrcoef

# statsmodels
import statsmodels.api as sm

# geography
import geopandas as gpd
import osmnx as ox
import shapely

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


#############
## metrics ##
#############

def plot_tf_performance(history):
    colors = ['darkgoldenrod', 'indianred']
    plt.figure(figsize=(10, 7), dpi=80)

    metrics = ['loss', 'accuracy', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[1], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.legend();
      
    
## confusion matrix
###################
def confusion_matrix_plot(conf_matrix):
    """ Confusion matrix matplotlib plot
    # param conf_matrix: nested list of TP, TN, FP, FN
    # return: None
    """
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.tight_layout()
    plt.show()
    
    
## roc auc
########## 
def plot_roc(fpr, tpr, roc_auc):
    ''''''
    plt.figure(figsize=(4, 5), dpi=80)
    lw = 2
    
    # plot ROC curve
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    
    # plot random guess
    plt.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=lw,
        linestyle="--",
        label= 'random guess'
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (Sensitivity)")
    plt.ylabel("True Positive Rate (1 - Specificity)")
    plt.title("Receiver operating characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.show()
    
    
    
## PR AUC
#########
def plot_prauc(prc, rec):
    ''''''
    plt.figure(figsize=(4, 5), dpi=80)
    lw = 2
    
    # plot ROC curve
    plt.plot(
        prc,
        rec,
        color="darkorange",
        lw=lw,
        label="PR AUC curve"
    )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Precision")
    plt.ylabel("Recal")
    plt.title("Area under the interpolated precision-recall curve (AUPRC)")
    plt.legend(loc="lower right")
    plt.show()
    
#####################   
## counterfactuals ##
##################### 
def test_counterfactuals(df, test_rlnI, test_rlnI_boolmat, test_x, test_y, test_y_boolmat, model_lstm):
    ''''''
    # pull test rlnI keys from df
    test_rlnI_nomask = test_rlnI[test_rlnI_boolmat == False]
    test_df_nomask = df[df.rlnI_key.isin(test_rlnI_nomask)]

    # keep only cols of interest
    cols = [
        'rlnI_updated', 'rlnI_key', 'visitsI_1yol',
        'age', 'M', 'white', 'black',
        'hisp', 'pm25I_hat',
        'outcome' 
    ]

    test_df_nomask = test_df_nomask[cols]

    # compute predicted outcome under different pm25I_hat values (counterfactuals)
    for idx, pm_level in enumerate([1, 0.99, 0.7, 0.3, 0]):
        print('Compute predictions for PM25Ihat level: ', pm_level)
        temp_test_x = test_x.copy()
        temp_test_y = test_y.copy()
        temp_test_x[:, :, 1:2] = temp_test_x[:, :, 1:2] * pm_level  # modify pm25 levels

        # run the trained model on the test data (the model outputs probabilities)
        temp_test_y_pred = model_lstm.predict(x=temp_test_x)

        ## Some of the predictions should be masked, and therefore removed from the final prediction dataset (similar to   
        #what we did above)
        temp_test_y_pp_nomask =  temp_test_y_pred[test_y_boolmat==False] # False removes masked values

        # transform probabilities to binary outcome
        temp_test_y_pred_nomask = np.where(temp_test_y_pp_nomask>=0.5, 1, 0)

        # add it to test_df_nomask
        test_df_nomask['outcome_pp_'+str(pm_level)] = temp_test_y_pp_nomask
        test_df_nomask['outcome_pred_'+str(pm_level)] = temp_test_y_pred_nomask
        
    return test_df_nomask

## counterfactual pm
####################
def plot_counterfactuals_pm(test_df_nomask):
    nrows, ncols = 2, 2
    f, axs = plt.subplots(nrows, ncols, figsize=(10,5))

    pm_level = ['0.99', '0.7', '0.3', '0']

    for idx, ax in enumerate(axs.flatten()):
        ax.hist(
            test_df_nomask['outcome_pp_1'],
            label = 'actual',
            density=True,
            alpha=0.5,
        )

        ax.hist(
            test_df_nomask['outcome_pp_'+pm_level[idx]],
            label = str(pm_level[idx] + '* PM25I_hat'),
            density=True,
            alpha=0.5,
        )

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        if idx in [2,3]:
            ax.set_xlabel('Probability of visit for resp/cardio')
        if idx in [0,2]:
            ax.set_ylabel('Density')

        diff_prob = (test_df_nomask['outcome_pp_1'].mean()- \
                     test_df_nomask['outcome_pp_'+pm_level[idx]].mean()) * 100 

        ax.set_title('Average of [P(actual) - P(counterfactual)]: ' + str(np.round(diff_prob,3)) + ' %')

        ax.legend(frameon=False)
    plt.tight_layout()
    
## counterfactual pm_race_sex
#############################    
def plot_counterfactuals_pm_race_sex(test_df_nomask):
    ''''''
    nrows, ncols = 2, 2
    f, axs = plt.subplots(nrows, ncols, figsize=(10,5))

    pm_level = [
        '1', '0.7',
        '1', '0.7'
    ]

    pm_level_label = [
        'actual', '0.7 * pm25I_hat',
        'actual', '0.7 * pm25I_hat'
    ]

    for idx, ax in enumerate(axs.flatten()):
        # for male vs. female
        if idx in [0,1]:
            # male
            temp_M = test_df_nomask[test_df_nomask.M.eq(1)]
            ax.hist(
                temp_M['outcome_pp_'+pm_level[idx]],
                label = pm_level_label[idx]+', male',
                density=True,
                alpha=0.5,
            )

            # female
            temp_F = test_df_nomask[test_df_nomask.M.eq(0)]
            ax.hist(
                temp_F['outcome_pp_'+pm_level[idx]],
                label = pm_level_label[idx]+', female',
                density=True,
                alpha=0.5,
            )

            diff_prob = (temp_M['outcome_pp_'+pm_level[idx]].mean() -\
                         temp_F['outcome_pp_'+pm_level[idx]].mean()) * 100 
            ax.set_title('Average of [P(male) -  P(female)]: ' + str(np.round(diff_prob,2)) + ' %')

        # for black vs white
        if idx in [2,3]:
            # black
            temp_B = test_df_nomask[test_df_nomask.black.eq(1)]
            ax.hist(
                temp_M['outcome_pp_'+pm_level[idx]],
                label = pm_level_label[idx]+', black',
                density=True,
                alpha=0.5,
            )

            # female
            temp_W = test_df_nomask[test_df_nomask.white.eq(0)]
            ax.hist(
                temp_F['outcome_pp_'+pm_level[idx]],
                label = pm_level_label[idx]+', white',
                density=True,
                alpha=0.5,
            )  


            diff_prob = (temp_B['outcome_pp_'+pm_level[idx]].mean() -\
                         temp_W['outcome_pp_'+pm_level[idx]].mean()) * 100 
            ax.set_title('Average of [P(black) -  P(white)]: ' + str(np.round(diff_prob,2)) + ' %')
            
        ax.legend(frameon=False, loc='upper left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if idx in [2,3]:
            ax.set_xlabel('Probability of visit for resp/cardio')
        if idx in [0,2]:
            ax.set_ylabel('Density')



    plt.tight_layout()
    
## counterfactual counts
########################  
def plot_counterfactuals_counts(test_df_nomask):
    ''''''
    # compute visit # reductions in counterfactual
    visit_reductions = [
        (1-test_df_nomask.outcome_pred_1.sum()/test_df_nomask.outcome.sum())*100,
        (1-test_df_nomask['outcome_pred_0.99'].sum()/test_df_nomask.outcome.sum())*100,
        (1-test_df_nomask['outcome_pred_0.7'].sum()/test_df_nomask.outcome.sum())*100,
        (1-test_df_nomask['outcome_pred_0.3'].sum()/test_df_nomask.outcome.sum())*100,
        (1-test_df_nomask['outcome_pred_0'].sum()/test_df_nomask.outcome.sum())*100,
    ]

    # add labels
    scenario = ['1 * PM25I_hat', '0.99 * PM25I_hat', '0.7 * PM25I_hat', '0.3 * PM25I_hat', '0 * PM25I_hat']

    # create df
    temp = pd.DataFrame({'visit_reductions':visit_reductions, 'scenario':scenario})

    # plot
    def add_value_label(x_list,y_list):
        for i in range(0, len(x_list)):
            ax.text(i,y_list[i],y_list[i], ha="center")
    scenario = temp.scenario
    change = np.round(temp.visit_reductions,2)
    
    plt.figure(figsize=(10,4))
    ax = plt.subplot(111)
    ax.bar(scenario, change)
    add_value_label(scenario,change)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Counterfactual predicted changes in visits for resp/cardio conditions\n (relative to observed)')
    ax.set_ylabel('percentage change (%)')
    ax.set_xlabel('counterfactual scenario')
    

    
## predictions by age and visits_1yol
######################################
def plot_counts_pred_vs_actual_age_1yol(test_df_nomask):
    ''''''
    # define features of interest
    features = ['age', 'visitsI_1yol']
    labels = ['age at visit time', 'number of visits 1st year of life']
    
    
    nrows, ncols = 1, 2
    f, axs = plt.subplots(nrows, ncols, figsize=(10,3))

    for idx, ax in enumerate(axs.flatten()):
        temp = test_df_nomask.groupby(features[idx], as_index=False).outcome.sum()
        temp['outcome'] = temp['outcome']/test_df_nomask.shape[0] *100

        temp_pred = test_df_nomask.groupby(features[idx], as_index=False).outcome_pred_1.sum()
        temp_pred['outcome_pred_1'] = temp_pred['outcome_pred_1']/test_df_nomask.shape[0] *100

        temp = temp.merge(temp_pred)

        ax.plot(temp[features[idx]], temp.outcome, label='actual')
        ax.plot(temp[features[idx]], temp.outcome_pred_1, label='predicted')
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel(labels[idx])
        ax.set_ylabel('Visits for resp/cardio (%)')
        ax.legend()
        
        
## predictions by age and visits_1yol
#####################################
def plot_counts_pred_vs_actual_demo(test_df_nomask):
    ''''''
    # males vs female predictions
    temp = test_df_nomask.groupby('M', as_index=False).outcome.sum()
    temp['outcome'] = temp['outcome']/test_df_nomask.shape[0] *100

    temp_pred = test_df_nomask.groupby('M', as_index=False).outcome_pred_1.sum()
    temp_pred['outcome_pred_1'] = temp_pred['outcome_pred_1']/test_df_nomask.shape[0] *100

    temp_M = temp.merge(temp_pred)
    temp_M['M'] = np.where(temp_M.M.eq(1), 'Male', 'Female')
    temp_M.rename(columns={'M':'demo'}, inplace=True)


    ## white vs. black predictions ##
    ################################
    # black
    temp = test_df_nomask.groupby('black', as_index=False).outcome.sum()
    temp['outcome'] = temp['outcome']/test_df_nomask.shape[0] *100

    temp_pred = test_df_nomask.groupby('black', as_index=False).outcome_pred_1.sum()
    temp_pred['outcome_pred_1'] = temp_pred['outcome_pred_1']/test_df_nomask.shape[0] *100
    temp_B = temp.merge(temp_pred)


    temp_B.rename(columns={'black':'demo'},inplace=True)
    temp_B['demo'] = np.where(temp_B.demo.eq(1), 'Black', 'others')
    temp_B = temp_B[temp_B.demo.eq('Black')]


    # white
    temp = test_df_nomask.groupby('white', as_index=False).outcome.sum()
    temp['outcome'] = temp['outcome']/test_df_nomask.shape[0] *100

    temp_pred = test_df_nomask.groupby('white', as_index=False).outcome_pred_1.sum()
    temp_pred['outcome_pred_1'] = temp_pred['outcome_pred_1']/test_df_nomask.shape[0] *100

    temp_W = temp.merge(temp_pred)
    temp_W

    temp_W.rename(columns={'white':'demo'},inplace=True)
    temp_W['demo'] = np.where(temp_W.demo.eq(1), 'White', 'others')
    temp_W = temp_W[temp_W.demo.eq('White')]

    temp_MBW = pd.concat([temp_M, temp_W, temp_B])

    ax = sns.barplot(
        x = 'demo',
        y = 'outcome',
        data = temp_MBW,
        color='#1f77b4',
        label='actual',
        alpha=0.5
    )

    ax = sns.barplot(
        x = 'demo',
        y = 'outcome_pred_1',
        data = temp_MBW,
        color='orange',
        label='predicted',        
        alpha=0.5
    )

    plt.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('Visits for resp/cardio (%)')
    
    
