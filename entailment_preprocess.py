# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 10:56:38 2021
preprocess entailment
@author: Liz
"""

import nltk
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import contractions
from string import punctuation
import numpy as np
from nltk import pos_tag_sents
from nltk import word_tokenize

import re


df_train=pd.read_csv('df_train._entail.csv')

df_test=pd.read_csv("df_test_entail.csv")

#train lowercase
df_train['Preprocess1'] = df_train['Sent1'].str.lower()
df_train["Preprocess2"]=df_train["Sent2"].str.lower()

#test lowercase
df_test['Preprocess1'] = df_test['Sent1'].str.lower()
df_test["Preprocess2"]=df_test["Sent2"].str.lower()

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

df_train["Preprocess1"]=df_train["Preprocess1"].apply(lambda x: replace_contractions(x))
df_train["Preprocess2"]=df_train["Preprocess2"].apply(lambda x: replace_contractions(x))

df_test["Preprocess1"]=df_test["Preprocess1"].apply(lambda x: replace_contractions(x))
df_test["Preprocess2"]=df_test["Preprocess2"].apply(lambda x: replace_contractions(x))

#print("PREPROCESS2", df["Preprocess"])


#strip punctuation
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

df_train["Preprocess1"]=df_train["Preprocess1"].apply(lambda x: strip_punctuation(x))
df_train["Preprocess2"]=df_train["Preprocess2"].apply(lambda x: strip_punctuation(x))

df_test["Preprocess1"]=df_test["Preprocess1"].apply(lambda x: strip_punctuation(x))
df_test["Preprocess2"]=df_test["Preprocess2"].apply(lambda x: strip_punctuation(x))

#STOPPED HERE
#df["Preprocess"]=df["Preprocess"].apply(lambda x: strip_punctuation(x))
#print("Preprocess3", df["Preprocess"])

#features=df["POS"].str.split()
#print("Preprocess4", df["Preprocess"])

def identify_tokens(row):
    #review = row['review']
    tokens = nltk.word_tokenize(row)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words
df["New_Preprocess"] = df['Preprocess'].str.split(" +",expand = False)
#tokenize Preprocess
df['New_Preprocess'] = df['Preprocess'].apply(identify_tokens)


#create new column for lemmatizing and lower case all text
df['Lemmatize'] = df['Preprocess']



lemmatizer = WordNetLemmatizer()
#
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

#apply POS tags
df['Lemmatize'] = df['Lemmatize'].apply(lambda v: nltk.pos_tag(nltk.word_tokenize(v)))

#lemmatize with POS
df['Lemmatize'] = df['Lemmatize'].transform(lambda value: ' '.join([lemmatizer.lemmatize(a[0],pos=get_wordnet_pos(a[1])) if get_wordnet_pos(a[1]) else a[0] for a in  value]))

def identify_tokens(row):
    #review = row['review']
    tokens = nltk.word_tokenize(row)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

#tokenize lemmatized words and take out numbers
df['Lemmatize'] = df['Lemmatize'].apply(identify_tokens)

#remove stop words
def remove_stopwords(text):
    words=[w for w in text if w not in stopwords.words('english')]
    return words


df["Lemmatize_No_Stop"]=df["Lemmatize"].apply(lambda x: remove_stopwords(x))

#print(df)
############################################################################
#create two more columns; one where all words stemmed; one where all stemmed
#and no stop words

##create new column for stemming and lower case all text
df['Stem'] = df['Preprocess']
#
##tokenize stemmed words and take out digits
df['Stem'] = df['Stem'].apply(identify_tokens)

stemmer=PorterStemmer()
df['Stem'] = df['Stem'].apply(lambda x: [stemmer.stem(y) for y in x]) 

#
df["Stem_No_Stop"]=df["Stem"].apply(lambda x: remove_stopwords(x))

##ADD POS TAG INFO


def posTag(data):
    data  = pd.DataFrame(df)
    sent = df['Preprocess'].tolist()
    #print (pos_tag_sents(map(word_tokenize, sent)))
    #ADD POS tags
    taggedComments =  pos_tag_sents(map(word_tokenize,  sent))
    #ADD POS tag to word so can treat different POS of same word as different word
    df['POS2'] = [' '.join([''.join(y) for y in x]) for x in taggedComments]
    #ADD just the POS by themselves
    df['POS'] = [' '.join([''.join(y[1]) for y in x]) for x in taggedComments]
    #add word POS tuples
    df["TUPLES"]=taggedComments
    return data


taggedData=posTag(df)

df['POS'] = df['POS'].apply(word_tokenize)  
df["POS2"]=df["POS2"].apply(word_tokenize)

        
#select certain words based on POS tag; remainder will just be POS TAG         
def TupleSplice(TUPLES):
    new=[]
    count=0
    for list in TUPLES:
        #determine if word meets the POS TAGS YOU ARE LOOKING FOR
        if list[1]=="IN" or list[1]=="VBD" or list[1]=="VBN" or list[1]=="VB" or list[1]=="VBP" or list[1]=="VBZ" or list[1]=="VBG" or list[1]=="PP":
            new.append(list[0])
            count=1+count
            if count==len(TUPLES)-1 and count>0:
                return new
        else:
            new.append(list[1])
            count=1+count
            if count==len(TUPLES)-1 and count>0:
                return new


def TupleSplice_WORD_ONLY(TUPLES):
    new=[]
    count=0
    for list in TUPLES:
        #print(list[1])
        #determine if word meets the POS TAGS YOU ARE LOOKING FOR
        if list[1]=="IN" or list[1]=="VBD" or list[1]=="VBN" or list[1]=="VB" or list[1]=="VBP" or list[1]=="VBZ" or list[1]=="VBG" or list[1]=="PP" or list[1]=="CC":
            new.append(list[0])
            count=1+count
            #print(count)
            if count==len(TUPLES)-1 and count>0:
                #print("TRUE")
                return new
        else:
            count=1+count
            #print(list[1])
            #print("FALSE")
            continue

df["VB_IN_PP_CC"]=df.apply(lambda x: TupleSplice(x["TUPLES"]), axis=1)

df["VB_IN_PP_WRD_ONLY"]=df.apply(lambda x: TupleSplice_WORD_ONLY(x["TUPLES"]), axis=1)


#drop any rows without any text in VB_IN_PP
df1 = df[df['VB_IN_PP_CC'].notna()]

#turn all words in list to single string with space between words
df1["VB_IN_PP_CC"]=df1["VB_IN_PP_CC"].apply(lambda x: ' '.join(map(str, x)))

#delete tuples column because unneeded for modesl
df1.drop(columns=['TUPLES'])


#df1.to_csv('stem_lem012121.csv', index=False)

df.replace(r'^\s*$', np.nan, regex=True)

#drop any rows without any text in VB_IN_PP
df = df[df['Preprocess'].notna()]
import re
df['Preprocess'] = df['Preprocess'].map(str)
#df['Label2'] = df['Label2'].map({"T": 0, "S": 0, "M": 0, 'N': 0, 'C': 1, "R": 0, "B":0, "U":0, "n": 0, "c": 1})

df.to_csv('stem_lem032121.csv', index=False)



#print(df)