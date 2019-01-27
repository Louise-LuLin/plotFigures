import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os,sys,re
import json
from itertools import cycle, islice

folder = './results/perplexity/'

# plot perplexity in normal setting
def plot_overall(length=50):
    x_range = np.arange(5, length+1, 5)
    x_ticks = [str(i) for i in x_range]
    sources=['yelp']
    source_label = ['Restaurants']
    models = ['ETBIR', 'LDA_Variational', 'CTM', 'ETBIR_Item', 'LDA_Item', 'RTM_item', 'ETBIR_User', 'LDA_User', 'RTM_user']
    labels = ['TUIR', 'LDA', 'CTM', 'iTUIR', 'iLDA', 'iRTM', 'uTUIR', 'uLDA', 'uRTM']
    modes = ['overall','0','1','2','3']
    styles = ['.-', '-', '-.', '--', '.-','--','-','-.','.-']

    fig, axes = plt.subplots(ncols=len(sources), nrows=1)
    ax = axes.flatten()
    for s in range(2):
        source = sources[s]
        mean = pd.DataFrame()
        var = pd.DataFrame()
        for i in range(len(models)):
            model=models[i]
            data = pd.read_csv(folder + source + '_70k_' + model + '.csv')
            mean[labels[i]] = data['mean']
            var[labels[i]] = data['var']
        # mean.set_index(['number_of_topics'], inplace=True)
        # var.set_index(['number_of_topics'], inplace=True)
            ax[s].errorbar(x_range, mean[labels[i]], yerr = var[labels[i]], fmt = styles[i], linewidth=2.0)
            # mean[labels[i]].plot(kind='line', ax=ax[s], yerr=var, fmt=styles[i], linewidth=2.0, legend = False)
        ax[s].set_title(source_label[s], fontsize=22)
        ax[s].set_xlabel("Number of Topics", fontsize=20)
        if s == 0:
            ax[s].set_ylabel("Perplexity", fontsize=21)
        ax[s].tick_params(axis = 'both', which = 'major', labelsize = 15)
        # ax[s].grid()
        # ax[s].set_xticklabels(x_ticks)
        if(s==0): # amazon perp scale
            ax[s].set_ylim([1000,2900])
        else: # yelp perp scale
            ax[s].set_ylim([600,1700])
        leg = ax[s].legend(loc = 'upper center', ncol=3, fancybox=True, fontsize = 13)
        leg.get_frame().set_alpha(0.7)
    # plt.subplots_adjust(hspace=0.5)
    fig.set_size_inches(11.5,5)
    # plt.tick_params(labelsize=20)
    plt.savefig('perp1.png', bbox_inches='tight')


# plot perplexity in cold start setting
def plot_separate(length=50):
    x_range = np.arange(5, length+1, 5)
    x_ticks = [str(i) for i in x_range]
    sources=['yelp']
    source_label = ['Restaurants']
    modes = ['cold1', 'cold2', 'cold3', 'cold4']
    mode_label = [r'$D^{Cold}_{u&i}$', r'$D^{Cold}_{u}$', r'$D^{Cold}_{i}$', r'$D^{Warm}$']

    models = ['LDA_Item', 'ETBIR_Item', 'LDA_User', 'ETBIR', 'ETBIR_User']
    labels = ['iLDA', 'iTUIR', 'uLDA', 'TUIR', 'uTUIR']
    styles = ['-', '--', '-.', ':', '-.']

    yelp_y_maxes = [1500, 1500]

    fig, axes = plt.subplots(ncols=len(modes), nrows=len(sources))
    for s in range(len(sources)):
        for m in range(len(modes)):
            source = sources[s]
            mode = modes[m]

            mean = pd.DataFrame()
            var = pd.DataFrame()
            for i in range(len(models)):
                model=models[i]
                data = pd.read_csv(folder + 'coldstart/' + mode + '_' + source + '_70k_' + model + '.csv')
                mean[labels[i]] = data['mean']
                var[labels[i]] = data['var']
            # mean.set_index(['number_of_topics'], inplace=True)
            # var.set_index(['number_of_topics'], inplace=True)
                axes[s][m].errorbar(x_range, mean[labels[i]], yerr = var[labels[i]], fmt = styles[i], linewidth=2.0)
                # mean[labels[i]].plot(kind='line', ax=axes[s][m], yerr=var, fmt=styles[i], linewidth=2.0, legend=False)
            axes[s][m].set_title(mode_label[m] + " of " + source_label[s], fontsize=20)
            if s == 1:
                axes[s][m].set_xlabel("Number of Topics", fontsize=16)
            axes[s][m].set_ylabel("Perplexity", fontsize=16)
            axes[s][m].tick_params(axis = 'both', which = 'major', labelsize = 12)
            # axes[s][m].grid()
            # axes[s][m].set_xticklabels(x_ticks)
            if(s==0):
                axes[s][m].set_ylim([800,2600])
            else:
                axes[s][m].set_ylim([550,1600])
            leg = axes[s][m].legend(loc = 'upper center', ncol=3, fancybox=True, fontsize=13)
            leg.get_frame().set_alpha(0.7)
    plt.subplots_adjust(hspace=0.4)
    fig.set_size_inches(26, 7)
    # plt.tick_params(labelsize=20)
    plt.savefig('perp2.png', bbox_inches='tight')

# plot_overall()
plot_separate()



