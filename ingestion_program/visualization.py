
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # seaborn for nice plot quicker
from sklearn.metrics import roc_curve
from IPython.display import display
from sklearn.metrics import roc_auc_score

class Dataset_visualise():

    def __init__(self, data, weights=None, labels=None, name="dataset"):
        self.dfall = data
        self.target = labels
        self.weights = weights
        self.columns = self.dfall.columns
        self.name = name



    def examine_dataset(self):
        print(f'[*] --- Dataset name : {self.name}')
        print(f'[*] --- Number of events : {self.dfall.shape[0]}')
        print(f'[*] --- Number of features : {self.dfall.shape[1]}')
        print(f'[*] --- Number of signal events : {self.dfall[self.target==1].shape[0]}')
        print(f'[*] --- Number of background events : {self.dfall[self.target==0].shape[0]}')

        print('[*] --- List of all features')
        for col in self.dfall.columns.values:
            print(f"\t{col}")

        print('[*] --- Examples of all features')
        display(self.dfall.head())

        print('[*] --- Description of all features')
        display(self.dfall.describe())

    def histogram_dataset(self,columns = None):
        fig = plt.figure()
        if columns == None:
            columns  =self.columns
        sns.set(rc={'figure.figsize':(40,40)})

        dfplot=pd.DataFrame(self.dfall, columns=columns)

        nbins = 50
        ax=dfplot[self.target==0].hist(weights=self.weights[self.target==0],figsize=(15,12),color='b',alpha=0.5,density=True, bins = nbins,label="B")
        ax=ax.flatten()[:dfplot.shape[1]] # to avoid error if holes in the grid of plots (like if 7 or 8 features)
        dfplot[self.target==1].hist(weights=self.weights[self.target==1],figsize=(15,12),color='r',alpha=0.5,density=True,ax=ax, bins = nbins,label="S")


        plt.legend(loc="best")
        plt.title('Histograms of features in' + self.name)
        plt.show()

    def correlation_plots(self,columns = None):
        caption = ["Signal feature","Background feature"]
        if columns == None:
            columns  =self.columns
        for i in range(2):
            sns.set(rc={'figure.figsize':(10,10)})

            dfplot=pd.DataFrame(self.dfall, columns=columns)

            print (caption[i]," correlation matrix")
            corrMatrix = dfplot[self.target==i].corr()
            sns.heatmap(corrMatrix, annot=True)
            plt.title('Correlation matrix of features in' + self.name)
            plt.show()


    def pair_plots(self,sample_size = 10,columns = None):
        if columns == None:
            columns  =self.columns
        df_sample = self.dfall[columns].copy()
        df_sample["Label"] = self.target
        
        df_sample_S = df_sample[self.target==1].sample(n=sample_size)
        df_sample_B = df_sample[self.target==0].sample(n=sample_size)
        frames = [df_sample_S,df_sample_B]
        del df_sample
        df_sample=pd.concat(frames)

        sns.set(rc={'figure.figsize':(16,14)})

        ax = sns.PairGrid(df_sample,hue = "Label")
        ax.map_upper(sns.scatterplot,alpha=0.3,size=0.3)
        ax.map_lower(sns.kdeplot,fill=True,levels=5,alpha=0.3)  # Change alpha value here
        ax.map_diag(sns.histplot,alpha=0.3)  # Change alpha value here
        ax.add_legend(title='Legend',labels=['Signal','Background'],fontsize=12)

        legend = ax._legend
        for lh in legend.legendHandles:
            lh.set_alpha(0.5)
            lh._sizes = [10]

        plt.rcParams['figure.facecolor'] = 'w'  # Set the figure facecolor to white
        ax.fig.suptitle('Pair plots of features in' + self.name)
        plt.show()
        plt.close()
        
    def cluster_plot(self,columns = None):
        
        plt.figure()

        if columns == None:
            columns  =self.columns
        df_sample = self.dfall[columns].copy()
        
        sns.clustermap(df_sample)


def Z_curve(score,labels,weights): ## work in progress
    
    plt.figure()

    
    sns.set(rc={'figure.figsize':(8,7)})

    thresholds_list =  np.linspace(0,1,num=100)
    int_pred_sig = [weights[(labels == 1) & (score  > th_cut)].sum() for th_cut in thresholds_list]
    plt.plot(thresholds_list,int_pred_sig)
    plt.show()

def roc_curve_(score,labels,weights,plot_label = "model",color='b',lw = 2):

    auc = roc_auc_score(y_true=labels, y_score=score,sample_weight=weights)

    
    plt.figure()

    plt.figure(figsize=(8,7))

    fpr,tpr,_ = roc_curve(y_true=labels, y_score=score,sample_weight=weights)
    plt.plot(fpr, tpr, color= color,lw=lw, label=plot_label + ' AUC :' + f'{auc:.3f}')

    plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.show()

    plt.close()

def events_histogram(score,labels,weights,plot_label = None,y_scale = 'log'):
    plt.figure()
    sns.set(rc={'figure.figsize':(8,7)})
    fig, ax = plt.subplots()


    high_low=(0,1)
    bins=30
    
    weights_signal = weights[labels == 1]
    weights_background = weights[labels == 0]

    plt.hist(score[labels == 1],
                 color='r', alpha=0.7,range=high_low, bins=bins,
                 histtype='stepfilled', density=False,
                 label='S',weights=weights_signal) # alpha is transparancy
    plt.hist(score[labels == 0],
                 color='b', alpha=0.7, range=high_low,  bins=bins,
                 histtype='stepfilled', density=False,
                 label='B', weights=weights_background)


    plt.legend()
    plt.title(plot_label)
    plt.xlabel(" Score ")
    plt.ylabel(" Number of events ")
    ax.set_yscale(y_scale)

    plt.show()
    plt.close()
    
def score_histogram(score,labels,plot_label=None,y_scale = 'log'):
    plt.figure()
    sns.set(rc={'figure.figsize':(8,7)})
    fig, ax = plt.subplots()


    high_low=(0,1)
    bins=30

    plt.hist(score[labels == 1],
                 color='r', alpha=0.7,range=high_low, bins=bins,
                 histtype='stepfilled', density=False,
                 label='S') # alpha is transparancy
    plt.hist(score[labels == 0],
                 color='b', alpha=0.7, range=high_low,  bins=bins,
                 histtype='stepfilled', density=False,
                 label='B')


    plt.legend()
    plt.title(plot_label)
    plt.xlabel(" Score ")
    plt.ylabel(" count ")
    ax.set_yscale(y_scale)

    plt.show()
    plt.close()
    
    
def validationcurve(results,eval_metric,model_name = 'model'):

    epochs = len(results['validation_0'][eval_metric])
    x_axis = range(0, epochs)
    plt.figure(figsize=(8,7))
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0'][eval_metric], label='Train')
    ax.plot(x_axis, results['validation_1'][eval_metric], label='Validation')
    ax.legend()
    plt.ylabel(eval_metric)
    plt.title(model_name + ' logloss')
    plt.show()

def feature_importance_plot (columns,feature_importance,model_name = 'model'):

    plt.figure(figsize=(8,7))

    plt.bar(columns, feature_importance)
    plt.xticks(rotation=90)
    plt.ylabel('Feature Importance')
    plt.title(model_name + ' Feature Importance')
    plt.show()

    top_features = []
    for feature, importance in zip(columns, feature_importance):
        if importance > 0.025:
            top_features.append(feature)
    
    print( top_features )

def permutation_importance(model,data,model_name = 'model'):
    
    plt.figure(figsize=(8,7))

    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, data.dfall, data.target,sample_weight=data.weights,
                            scoring='roc_auc',n_repeats=1,n_jobs=-1,
                            random_state=0)
    plt.bar(data.columns,r.importances.mean(axis=1).T,)

    plt.xlabel('features')
    plt.xticks(rotation=90)
    plt.ylabel('impact on auc')
    plt.title('Permutation Importance XGBoost + ' + model_name)

    plt.show()

