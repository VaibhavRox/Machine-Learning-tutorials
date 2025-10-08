#Algorithmic implementation of decision tree classifier

import pandas as pd
import math

def entropy(column):
    values=column.unique()
    total=len(column)
    ent=0
    for val in values:
        p=len(column[column==val])/total
        ent+=-p*math.log2(p)
    return ent

def ig(data,feature,target):
    total_ent=entropy(data[target])
    values=data[feature].unique()
    weighted_ent=0
    for val in values:
        subset=data[data[feature]==val]
        weighted_ent+=(len(subset)/len(data))*entropy(subset[target])
    return total_ent-weighted_ent

def id3(data,features,target):
    #if all target values same, return that
    if len(data[target].unique())==1:
        return data[target].iloc[0]
    #if no features left, return majority class
    if len(features)==0:
        return data[target].mode()[0]
    #now find max ig feature
    best_col=None
    best_gain=-1
    for f in features:
        gain=ig(data,f,target)
        if gain>best_gain:
            best_gain=gain
            best_col=f
    tree={best_col:{}}

    for val in data[best_col].unique():
        subset=data[data[best_col]==val]
        new_features=[f for f in features if f!=best_col]
        subtree=id3(subset,new_features,target)
        tree[best_col][val]=subtree
    return tree

def predict(tree,sample):
    if not isinstance(tree,dict):
        return tree
    col=list(tree.keys())[0]
    value=sample.get(col)
    if value not in tree[col]:
        return None
    return predict(tree[col][value],sample)

def print_tree(tree,indent=' '):
    if not isinstance(tree,dict):
        print(f"{indent}: {tree}")
        return
    feature=next(iter(tree))
    print(f"{indent}{feature}")
    for val,subtree in tree[feature].items():
        print(f"{indent}|--{val}",end="")
        if isinstance(subtree,dict):
            print()
            print_tree(subtree,indent+'| ')
        else:
            print(f": {subtree}")

df=pd.read_csv('salaries.csv')
target=df.columns[-1]
features = list(df.columns[:-1])

dec_tree=id3(df,features,target)
print(dec_tree)

print_tree(dec_tree)
