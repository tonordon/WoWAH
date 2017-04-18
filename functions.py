import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve. All credit for this function goes to the 
    scikit-learn team. Code was swiped from here: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

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
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
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

def distill_wowah(wow_df, chars, MAX_LEVEL1=70, MAX_LEVEL2=80, WLK_RD=pd.to_datetime('11/18/2008')):
    """
    Uses the pandas groupby and agg functions to distill the WoWAH dataset into one dataframe with average quantities for each caharacter.
    
    Parameters
    ----------
    df: The WoWAH pandas dataframe
    chars: List of unique character IDs in WoWAH dataset.
    
    Returns
    --------
    av_df: The avatar distilled dataframe.
    """

    #Make the new dataframe for progression quantities
    av_df = pd.DataFrame({'char': chars}).sort_values(by='char')

    #Note that as another exploration for this dataset has found:
    #(https://www.kaggle.com/romainvincent/d/mylesoneill/warcraft-avatar-history/exploration)
    # that here seems to be several avatars that mysteriously change race throughout the observations.
    #So we want to also calculate the number of unique races and classes for each avatar as well
    #Number of races per avatar
    av_df = av_df.merge(wow_df[['char', 'race']].groupby(['char'], as_index=False).agg(lambda x: len(x.unique())), 
                on='char')
    av_df = av_df.rename(columns={'race':'nrace'})
    #Number of classes per avatar
    av_df = av_df.merge(wow_df[['char', 'charclass']].groupby(['char'], as_index=False).agg(lambda x: len(x.unique())), 
                on='char')
    av_df = av_df.rename(columns={'charclass':'ncharclass'})


    #Inclue the race and character class
    av_df = av_df.merge(wow_df[['char', 'race']].groupby(['char'], as_index=False).agg(
            lambda x: x.value_counts().index[0]), on='char')
    av_df = av_df.merge(wow_df[['char', 'charclass']].groupby(['char'], as_index=False).agg(
            lambda x: x.value_counts().index[0]), on='char')

    #Get mean levels and merge with av_df
    av_df = av_df.merge(wow_df[['char', 'level']].groupby(['char'], as_index=False).mean(), on='char')
    #And rename column to be more appropriate
    av_df = av_df.rename(columns={'level':'avglvl'})

    #Max level
    av_df = av_df.merge(wow_df[['char', 'level']].groupby(['char'], as_index=False).max(), on='char')
    #And rename column to be more appropriate
    av_df = av_df.rename(columns={'level':'maxlvl'})

    #Reach max level?
    av_df = av_df.merge(wow_df[['char', 'level']].groupby(['char'], as_index=False)\
                        .agg(lambda x: np.size(np.where(x >= MAX_LEVEL2)[0]) > 0), on='char')
    #And rename column to be more appropriate
    av_df = av_df.rename(columns={'level':'maxlvld'})

    #Reach max level prior to WLK?
    av_df = av_df.merge(wow_df.loc[wow_df['timestamp'] < WLK_RD, ['char', 'level']].groupby(['char'], as_index=False)\
                        .agg(lambda x: np.size(np.where(x >= MAX_LEVEL1)[0]) > 0), how='left', on='char')
    #And rename column to be more appropriate
    av_df = av_df.rename(columns={'level':'maxlvld_preWLK'})

    #Level range
    av_df = av_df.merge(wow_df[['char', 'level']].groupby(['char'], as_index=False)\
                        .agg(lambda x: x.max() - x.min()), on='char')
    #And rename column to be more appropriate
    av_df = av_df.rename(columns={'level':'lvlrng'})

    #Number unique of guilds
    av_df = av_df.merge(wow_df[['char', 'guild']].groupby(['char'], as_index=False).agg(lambda x: len(x.unique())), 
                on='char')
    av_df = av_df.rename(columns={'guild':'nguild'})

    #Most frequented guild
    av_df = av_df.merge(wow_df[['char', 'guild']].groupby(['char'], as_index=False).agg(lambda x: x.value_counts().index[0]),
                on='char')
    av_df = av_df.rename(columns={'guild':'modguild'})

    #Most frequented location
    av_df = av_df.merge(wow_df[['char', 'zone']].groupby(['char'], as_index=False).agg(lambda x: x.value_counts().index[0]), 
                on='char')
    av_df = av_df.rename(columns={'zone':'modzon'})

    #Number unique of locations
    av_df = av_df.merge(wow_df[['char', 'zone']].groupby(['char'], as_index=False).agg(lambda x: len(x.unique())), 
                on='char')
    av_df = av_df.rename(columns={'zone':'nzon'})

    #Number of data points for this avatar (i.e. frequency of playing)
    av_df = av_df.merge(pd.DataFrame(wow_df[['char', 'timestamp']].groupby(['char'], as_index=False).count()), on='char')
    av_df = av_df.rename(columns={'timestamp':'nplays'})

    #Last recorded play timestamp
    av_df = av_df.merge(wow_df[['char', 'timestamp']].groupby(['char'], as_index=False).max(), on='char')
    av_df = av_df.rename(columns={'timestamp':'lastplay'})

    #First recorded play timestamp
    av_df = av_df.merge(wow_df[['char', 'timestamp']].groupby(['char'], as_index=False).min(), on='char')
    av_df = av_df.rename(columns={'timestamp':'firstplay'})

    #And total time baseline
    av_df = av_df.merge(wow_df[['char', 'timestamp']].groupby(['char'], as_index=False)\
                        .agg(lambda x: x.max() - x.min()), on='char')
    av_df = av_df.rename(columns={'timestamp':'baseline'})
    #Convert this time delta to just a float number of days
    av_df['baseline_td'] = av_df['baseline'].dt.total_seconds() / (24 * 60 * 60)

    #Also want to know time spent actually progressing before max level was reached
    av_df = av_df.merge(wow_df.loc[wow_df['level'] < MAX_LEVEL2, 
                                   ['char', 'timestamp']].groupby(['char'], as_index=False)\
                        .agg(lambda x: x.max() - x.min()), on='char')
    av_df = av_df.rename(columns={'timestamp':'prog_baseline'})
    av_df['prog_baseline_td'] = av_df['prog_baseline'].dt.total_seconds() / (24 * 60 * 60)

    #Also, want to know if avatar stopped playing before Wrath of the Lich King expansion was released
    av_df = av_df.merge(wow_df[['char', 'timestamp']].groupby(['char'], as_index=False)\
                        .agg(lambda x: np.size(np.where(x >= WLK_RD)[0]) <= 0), on='char')
    av_df = av_df.rename(columns={'timestamp':'preWLK'})

    #Also, want to know if avatar started playing after Wrath of the Lich King expansion was released
    av_df = av_df.merge(wow_df[['char', 'timestamp']].groupby(['char'], as_index=False)\
                        .agg(lambda x: np.size(np.where(x < WLK_RD)[0]) <= 0), on='char')
    av_df = av_df.rename(columns={'timestamp':'postWLK'})
    
    return av_df
