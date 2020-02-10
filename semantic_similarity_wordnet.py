# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 14:51:22 2018

@author: Lin Wang
"""
## This is to check words' similarity with WordNet
import csv
import random
import numpy as np

import nltk
nltk.download('wordnet')
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

#Load an information content file from the wordnet_ic corpus
nltk.download('wordnet_ic')
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

#Or you can create an information content dictionary from a corpus (or anything that has a words() method)
from nltk.corpus import genesis
genesis_ic = wn.ic(genesis, False, 0.0)


'''
###########################################################
######### examples to look up pair-wise similarity
###########################################################
#To get the sense of the target word based on the context
sent = ['I', 'went', 'to', 'the', 'bank', 'to', 'deposit', 'money', '.']
print(lesk(sent, 'bank', 'n'))
wn.synset('savings_bank.n.02')

#To check the definition of a particular sense
wn.synset('buy.v.01').definition()

#To get all the senses of a particular word
for i in wn.synsets('soften'):
    print(str(i), i.definition())


#To get the similarity between two words: e.g. Resnik
wn.synset('blindfold.v.01').res_similarity(wn.synset('warn.v.01'), brown_ic)

# path similarity
wn.path_similarity(wn.synset('judge.n.01'), wn.synset('horse.n.01'))

# Leacock-Chodorow Similarity
wn.lch_similarity(wn.synset('judge.n.01'), wn.synset('horse.n.01'))

# Jiang-Conrath Similarity
wn.jcn_similarity(wn.synset('judge.n.01'), wn.synset('horse.n.01'), brown_ic)

#For more functions, see http://www.nltk.org/howto/wordnet.html'''


###################################################################
## calculate the item pair-wise correlations for different measures
###################################################################
cond = 'inanimate'
ani_list = []
with open('Noun_'+cond+'_list.txt', newline='', encoding = "Latin-1") as inputfile:
    for row in csv.reader(inputfile, delimiter=' '):
        ani_list.append(row)

# Resnik Similarity
results_res = np.zeros((len(ani_list),len(ani_list)))
word_pairs = []
for indrow in range(0,len(ani_list)):
    print("word: " + ani_list[indrow][0][8:-2])
    for indcolum in range(0,len(ani_list)):
        print(ani_list[indrow][0][8:-2] + " vs. " + ani_list[indcolum][0][8:-2])
        results_res[indrow][indcolum]=wn.synset(ani_list[indrow][0][8:-2]).res_similarity(wn.synset(ani_list[indcolum][0][8:-2]),brown_ic)
        if results_res[indrow][indcolum] == 1:
            word_pairs.append(ani_list[indrow][0] + " vs. " + ani_list[indcolum][0])
print(results_res)

#Jiang-Conrath Similarity
results_jcn = np.zeros((len(ani_list),len(ani_list)))
word_pairs = []
for indrow in range(0,len(ani_list)):
    print("word: " + ani_list[indrow][0][8:-2])
    for indcolum in range(0,len(ani_list)):
        print(ani_list[indrow][0][8:-2] + " vs. " + ani_list[indcolum][0][8:-2])
        results_jcn[indrow][indcolum]=wn.jcn_similarity(wn.synset(ani_list[indrow][0][8:-2]), wn.synset(ani_list[indcolum][0][8:-2]),brown_ic)
        if results_jcn[indrow][indcolum] == 1:
            word_pairs.append(ani_list[indrow][0] + " vs. " + ani_list[indcolum][0])
print(results_jcn)

#Lin Similarity
results_lin = np.zeros((len(ani_list),len(ani_list)))
word_pairs = []
for indrow in range(0,len(ani_list)):
    print("word: " + ani_list[indrow][0][8:-2])
    for indcolum in range(0,len(ani_list)):
        print(ani_list[indrow][0][8:-2] + " vs. " + ani_list[indcolum][0][8:-2])
        results_lin[indrow][indcolum]=wn.lin_similarity(wn.synset(ani_list[indrow][0][8:-2]), wn.synset(ani_list[indcolum][0][8:-2]),brown_ic)
        if results_lin[indrow][indcolum] == 1:
            word_pairs.append(ani_list[indrow][0] + " vs. " + ani_list[indcolum][0])
print(results_lin)

#path similarity
results_path = np.zeros((len(ani_list),len(ani_list)))
word_pairs = []
for indrow in range(0,len(ani_list)):
    print("word: " + ani_list[indrow][0][8:-2])
    for indcolum in range(0,len(ani_list)):
        print(ani_list[indrow][0][8:-2] + " vs. " + ani_list[indcolum][0][8:-2])
        results_path[indrow][indcolum]=wn.path_similarity(wn.synset(ani_list[indrow][0][8:-2]), wn.synset(ani_list[indcolum][0][8:-2]))
        if results_path[indrow][indcolum] == 1:
            word_pairs.append(ani_list[indrow][0] + " vs. " + ani_list[indcolum][0])
print(results_path)


#Leacock-Chodorow Similarity
results_lch = np.zeros((len(ani_list),len(ani_list)))
word_pairs = []
for indrow in range(0,len(ani_list)):
    print("word: " + ani_list[indrow][0][8:-2])
    for indcolum in range(0,len(ani_list)):
        print(ani_list[indrow][0][8:-2] + " vs. " + ani_list[indcolum][0][8:-2])
        results_lch[indrow][indcolum]=wn.lch_similarity(wn.synset(ani_list[indrow][0][8:-2]), wn.synset(ani_list[indcolum][0][8:-2]))
        if results_lch[indrow][indcolum] == 1:
            word_pairs.append(ani_list[indrow][0] + " vs. " + ani_list[indcolum][0])
print(results_lch)


#Wu-Palmer Similarity
results_wup = np.zeros((len(ani_list),len(ani_list)))
word_pairs = []
for indrow in range(0,len(ani_list)):
    print("word: " + ani_list[indrow][0][8:-2])
    for indcolum in range(0,len(ani_list)):
        print(ani_list[indrow][0][8:-2] + " vs. " + ani_list[indcolum][0][8:-2])
        results_wup[indrow][indcolum]=wn.wup_similarity(wn.synset(ani_list[indrow][0][8:-2]), wn.synset(ani_list[indcolum][0][8:-2]))
        if results_wup[indrow][indcolum] == 1:
            word_pairs.append(ani_list[indrow][0] + " vs. " + ani_list[indcolum][0])
print(results_wup)


###################################################################
## calculate the pair-wise correlations between different measures
###################################################################
import pandas as pd
import matplotlib.pyplot as plt

##correlate the whole matrix (including diagonal line)
df = pd.DataFrame({'res': np.concatenate(results_res)})
df['jcn'] = np.concatenate(results_jcn)
df['lin'] = np.concatenate(results_lin)
df['path'] = np.concatenate(results_path)
df['lch'] = np.concatenate(results_lch)
df['wup'] = np.concatenate(results_wup)

df.corr()

##plot the correlations in a matrix
plt.matshow(df.corr())
plt.xticks(range(len(df.columns)), df.columns)
plt.yticks(range(len(df.columns)), df.columns)
plt.colorbar()
plt.show()


#######################################################################
## calculate the whole similarity in python: include all words
#######################################################################
#read the list of words: all nouns
conds = ['animate','inanimate']
word_list = []
for cond in conds:
    with open('Noun_'+cond+'_list.txt', newline='', encoding = "Latin-1") as inputfile:
        for row in csv.reader(inputfile, delimiter=' '):
            word_list.append(row)

#build the similarity matrix
results = np.zeros((len(word_list),len(word_list)))
word_pairs = []
for indrow in range(0,len(word_list)):
    print("word: " + word_list[indrow][0][8:-2])
    for indcolum in range(0,len(word_list)):
        print(word_list[indrow][0][8:-2] + " vs. " + word_list[indcolum][0][8:-2])
        results[indrow][indcolum]=wn.wup_similarity(wn.synset(word_list[indrow][0][8:-2]), wn.synset(word_list[indcolum][0][8:-2]))
        #results[indrow][indcolum]=wn.path_similarity(wn.synset(word_list[indrow][0][8:-2]), wn.synset(word_list[indcolum][0][8:-2]))
        #results[indrow][indcolum]=wn.lch_similarity(wn.synset(word_list[indrow][0][8:-2]), wn.synset(word_list[indcolum][0][8:-2]))
        if results[indrow][indcolum] == 1:
            word_pairs.append(word_list[indrow][0] + " vs. " + word_list[indcolum][0])
np.save("Wu_AllNouns_similarity",results)
np.save("Wu_AllNouns_repeated",word_pairs)


#read the list of words: all verbs
conds = ['animate','inanimate']
word_list = []
for cond in conds:
    with open('Verb_'+cond+'_list.txt', newline='', encoding = "Latin-1") as inputfile:
        for row in csv.reader(inputfile, delimiter=' '):
            word_list.append(row)

#build the similarity matrix
results = np.zeros((len(word_list),len(word_list)))
word_pairs = []
for indrow in range(0,len(word_list)):
    print("word: " + word_list[indrow][0][8:-2])
    for indcolum in range(0,len(word_list)):
        print(word_list[indrow][0][8:-2] + " vs. " + word_list[indcolum][0][8:-2])
        #results[indrow][indcolum]=wn.wup_similarity(wn.synset(word_list[indrow][0][8:-2]), wn.synset(word_list[indcolum][0][8:-2]))
        #results[indrow][indcolum]=wn.path_similarity(wn.synset(word_list[indrow][0][8:-2]), wn.synset(word_list[indcolum][0][8:-2]))
        results[indrow][indcolum]=wn.lch_similarity(wn.synset(word_list[indrow][0][8:-2]), wn.synset(word_list[indcolum][0][8:-2]))
        if results[indrow][indcolum] == 1:
            word_pairs.append(word_list[indrow][0] + " vs. " + word_list[indcolum][0])
np.save("Lch_AllVerbs_similarity",results)
np.save("Wu_AllVerbs_repeated",word_pairs)


#######################################################################
## visualize the similarity matrix
#######################################################################
#plot the similarity matrix: all nouns
import pandas as pd
import matplotlib.pyplot as plt

results = np.load("Wu_AllNouns_similarity.npy")
#results = np.load("Lch_AllNouns_similarity.npy")
#results = np.load("path_AllNouns_similarity.npy")

# use imshow
plt.imshow(results);
plt.clim(0,3)
plt.colorbar()
plt.show()
plt.xticks(np.arange(0, len(df.columns), step=50))
plt.yticks(np.arange(0, len(df.columns), step=50))


#######################################################################
## stats from results that include all words
#######################################################################
results = np.load("Lch_AllNouns_similarity.npy")

ani = results[0:348,0:348]
ani[ani == 1] = np.nan
#ani[ani >= 3] = np.nan #for Lch
reform_ani = ani[np.triu_indices(len(ani))] #get the upper half of matrix
ani_sel = reform_ani[~np.isnan(reform_ani)]
means_ani = np.nanmean(ani_sel)
std_ani = np.nanstd(ani_sel)

inani = results[348:700,348:700]
inani[inani == 1] = np.nan
#inani[inani >= 3] = np.nan #for Lch
reform_inani = inani[np.triu_indices(len(inani))] #get the upper half of matrix
inani_sel = reform_inani[~np.isnan(reform_inani)]
means_inani = np.nanmean(inani_sel)
std_inani = np.nanstd(inani_sel)
print("animate="+str(means_ani) + "(" + str(std_ani) + "); inaimate="+str(means_inani) + "(" + str(std_inani) + ")")


#permutations
Perm=[]
for nperm in range(0,1000):
    new = np.concatenate([ani_sel,inani_sel])
    random.shuffle(new)
    rand_ani = new[0:len(ani)]
    rand_inani = new[-len(inani):]
    mean_rand_ani = np.nanmean(rand_ani)
    mean_rand_inani = np.nanmean(rand_inani)
    Perm.append(mean_rand_ani - mean_rand_inani)
cluster_pval = (np.sum(np.abs(Perm) > np.abs(ori_dif))+1)/np.float(nperm+1)
print("animate="+str(means_ani) + "(" + str(std_ani) + "); inaimate="+str(means_inani) + "(" + str(std_inani) + "); cluster_pval="+str(cluster_pval))
