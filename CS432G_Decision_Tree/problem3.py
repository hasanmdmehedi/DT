import numpy as np
import matplotlib.pyplot as plt
import math
from math import log
import pandas as pd
import random
from random import randrange
import csv
from csv import reader
from pprint import pprint
import string


def check_common(data):
    #print(data)
    label_col = data[:, -1]
    #print('label_col : ', label_col)
    unique_cls = np.unique(label_col)
    if len(unique_cls) == 1:
        return True
    else:
        return False


def data_classify(df):
    label_col = df[:, -1]
    unique_classes, cnts_unique_classes = np.unique(label_col, return_counts = True)
    index = cnts_unique_classes.argmax()
    classification = unique_classes[index]
    return  classification



def data_splits(data):
    #print(data)
    tentative_splits = {}
    n_rows, n_columns = data.shape
    for i in range(n_columns - 1):
        tentative_splits[i] = []
        values = data[:, i]
        #print(values)
        unique_values = np.unique(values)
        #print(unique_values)
        for j in range(len(unique_values)):
            if j != 0:
                #print(unique_values[j])
                tentative_split = (unique_values[j] + unique_values[j-1])/2
                tentative_splits[i].append(tentative_split)
    return tentative_splits




def split_data_feature(df, split_column, split_value):
    return df[df[:, split_column] <= split_value], df[df[:, split_column] >= split_value]



def calEnt(data):
    numEntries = len(data)
    labelCnts = {}
    for i in data:
        currentlabel = i[-1]
        if currentlabel not in labelCnts.keys():
            labelCnts[currentlabel] = 0
        labelCnts[currentlabel] += 1
    Entropy = 0.0
    for j in labelCnts:
        p = float(labelCnts[j]) / numEntries
        Entropy = Entropy - p * log(p, 2)
    return Entropy



def ID3_choose_best_split(data, potential_splits):
    overall_entropy = 1
    gain_before = calEnt(data)
    #print(gain_before)                                                                               

    bestInformationGain = 0.0
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_1s, data_2s = split_data_feature(data, split_column=column_index, split_value=value)
            #print(len(np.unique(data[:, column_index])))
            n = len(data_1s) + len(data_2s)
            p_data_1s = len(data_1s) / n
            p_data_2s = len(data_2s) / n
            Ent_spl = (p_data_1s * calEnt(data_1s)
                          + p_data_2s * calEnt(data_2s))
            gain = gain_before - Ent_spl
            #print(gain)                                                                             

            if gain > bestInformationGain:
                bestInformationGain = gain
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value


# this part is partly taken from https://www.sebastian-mantey.com/code.html
def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5):    
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df           
        
    # base cases
    if (check_common(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = data_classify(data)
        
        return classification

   
    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = data_splits(data)
        split_column, split_value = ID3_choose_best_split(data, potential_splits)
        data_below, data_above = split_data_feature(data, split_column, split_value)
        
        # instantiate sub-tree
        feature_name = COLUMN_HEADERS[split_column]
        question = "{} <= {}".format(feature_name, split_value)
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)

        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree


# taken from https://www.sebastian-mantey.com/code.html 
def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))
    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)    
    return train_df, test_df



    
# this part is partly taken from https://www.sebastian-mantey.com/code.html
def classify_example(example, tree):
    question = list(tree.keys())[0]
    #print('question : ', question)
    feature_name, comparison_operator, value = question.split()

    #print(feature_name)
    if example[feature_name] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)


# this part is partly taken from https://www.sebastian-mantey.com/code.html
def calculate_accuracy(df, tree):
    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))
    df["classification_correct"] = df["classification"] == df["label"]
    accuracy = df["classification_correct"].mean()
    return accuracy



def randStr(chars = string.ascii_uppercase + string.digits, N=10):
	return ''.join(random.choice(chars) for _ in range(N))


def process_data(raw_data, Legend_L):
    raw_data = raw_data.assign(Legendary = '5')
    row_d, col_d = raw_data.shape
    legendset = np.array(Legend_L, dtype = float)
    dataset = np.array(raw_data, dtype = float)
    for i in range(row_d):
        if (legendset[i][0]==False):
            #print('False in loop')                                                                     
            #print(Legend_L.values[i][0])                                                               
            dataset[i][col_d-1]=0
        else:
            dataset[i][col_d-1]=1.0
            #print(Legend_L.values[i][0])                                                               
            #print(True)                                                                                
            #print(raw_data.values[i][col_d-1])                                                         
    return dataset


raw_data=pd.read_csv('data/pokemonStats.csv')
Legend_L = pd.read_csv('data/pokemonLegendary.csv')
labels = raw_data.columns.values
#print(labels)
dataset = process_data(raw_data, Legend_L)
dataset = pd.DataFrame(dataset)
row_d, col_d = dataset.shape

#print(dataset)
#Renaming title of the data
rand_l = []
for i in range(col_d-1):
    rand_l.append(randStr(N=5))

for i in range(col_d-1):
    dataset = dataset.rename(columns={i: rand_l[i]})
dataset = dataset.rename(columns={col_d-1: "label"})

#print(rand_l)

#random.seed()
train_df, test_df = train_test_split(dataset, test_size=400)
tree = decision_tree_algorithm(train_df, max_depth=3)
#pprint(tree)

example = test_df.iloc[1]
#print(example)
#print(classify_example(example, tree))

accuracy = calculate_accuracy(test_df, tree)
print('Regular Accuracy is', accuracy*100 )

#print(potential_splits)
