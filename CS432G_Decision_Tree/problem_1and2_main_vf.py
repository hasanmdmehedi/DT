import numpy as np
import matplotlib.pyplot as plt
import math
from math import log
import pandas as pd
import random
from random import randrange
from csv import reader
from pprint import pprint

def read_dataset(filename, split):
    fr=open(filename,'r')
    all_lines=fr.readlines()
    labels=['feature1', 'feature2', 'label']
    labelCounts={}
    dataset=[]
    for line in all_lines[0:]:
        line=line.strip().split(',')
        dataset.append(line)
    test = list()
    test_size = round(split * len(dataset))
    dataset_copy = list(dataset)
    while len(test) < test_size:
        #print(dataset_copy)
        index = randrange(len(dataset_copy))
        #print(index)
        test.append(dataset_copy.pop(index))
        #print(train)
    return (dataset_copy),labels,(test), dataset

def check_common(data):
    #print(data)
    label_col = data[:, -1]
    #print('label col : ', label_col)
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


def id3_DT_algorithm(df, labels, counter=0, min_samples=2, max_depth=3):
    global Label
    #print(labels)
    if counter == 0:
        Label = labels
        data = df
        #print(labels)
    else:
        data = df
    # base cases
    if check_common(data):
        class_f = data_classify(data)
        return class_f

    # recursive part
    else:
        counter += 1
        # helper
        #print(labels)
        tentative_splits = data_splits(data)
        split_column, split_value = ID3_choose_best_split(data, tentative_splits)
        #print(split_column)
        #print(split_value)
        data_below, data_above = split_data_feature(data, split_column, split_value)

        # sub-tree
        feature_name = Label[split_column]
        decision = "{} <= {}".format(feature_name, split_value)
        sub_tree = {decision: []}
        p_ans = id3_DT_algorithm(data_below, counter, max_depth)
        n_ans = id3_DT_algorithm(data_above, counter, max_depth)
        sub_tree[decision].append(p_ans)
        sub_tree[decision].append(n_ans)
        #print(split_value)
        return sub_tree

def prediction(example, tree):
    #print('ins clas : ', example[2])
    question = list(tree.keys())[0]
    feature_n, _, value = question.split()
    if feature_n == 'feature1':
        ii = 0
    else:
        ii = 1
    if example[ii] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]

    if not isinstance(answer, dict):
        return answer
    else:
        residual_tree = answer
        return prediction(example, residual_tree)


def calculate_accuracy(df, tree):
    #print(df)
    a = []
    res=0
    for i in range(0,len(df)):
        example = df.iloc[i]
        #print(example)
        #print('classify example : ', classify_example(example, tree))
        b = prediction(example, tree)
        a.append(b)
        #a[i].append(classify_example(example, tree))
    for	j in range(0,len(df)):
        if (int(a[j])==int(df.values[j][2])):
            res= res+1.0
        else:
            res= res +0.0
    accuracy = res/len(df)
    return accuracy


def cross_test_train(folds, cv, n_folds):
    train_cross = []
    kk = 0
    for i in range(0, n_folds):
        #print(i)
        #train_cross = []
        if (i == cv):
            test_cross= folds[i]
        else:
            n = len(folds[i])
            kk=kk+n
            #print(n)
            for k in range(0, n):
                b = folds[i][k]
                for m in range(0,3):
                    #print(b[0])
                    train_cross.append(b[m])
    train_cross = np.resize(train_cross,(kk,3))
    return test_cross, train_cross



def plot_data(df, title):
    row, col = df.shape
    #print(row)
    global original_data
    original_data = df
    features = []
    for i in range(len(df[0]) - 1):
        features.append(i)
    tree  = id3_DT_algorithm(df, labels)
    #pprint(tree)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    assign_data = np.asarray([[1, 1], [1, 2], [1, 3], [1, 4],[2, 1], [2, 2], [2, 3], [2, 4],
                            [3, 1], [3, 2], [3, 3], [3, 4], [4, 1], [4, 2], [4, 3], [4, 4]])

    ax1.scatter(df[:round(row/2)-1, 0], df[:round(row/2)-1, 1], s=5, c='b', marker="o")
    ax1.scatter(df[round(row/2):, 0], df[round(row/2):, 1], s=5, c='y', marker="o")
    title = 'synthetic-' + str(title)
    plt.title(title)
    plt.xlabel('feature')
    plt.ylabel('feature')
    plt.show()



    


if __name__ == '__main__': 
    filename='data/synthetic-1.csv'
    trainset, labels, testset, dataset = read_dataset(filename, split = 0.2)
    trainset = np.array(trainset, dtype = float)
    testset = np.array(testset, dtype = float)
    tree = id3_DT_algorithm(trainset, labels)
    testset = pd.DataFrame(testset)
    
    accuracy_r = calculate_accuracy(testset, tree)
    print('Regular Accuracy is', accuracy_r*100 )
    #print(testset)

    
    dataset=pd.DataFrame(dataset)
    tot_row = dataset.shape[0]
    dataset = dataset.head(tot_row).sample(n=tot_row, replace=False)

    n_folds=5
    dataset = np.array(dataset, dtype = float)
    new_data = np.array_split(dataset, n_folds)
    #print('test_cross : ', new_data[0])
    #print(labels)
    for i in range(0, n_folds):
        cv = i
        test_cross, train_cross = cross_test_train(new_data, cv, n_folds)
        tree_cross = id3_DT_algorithm(train_cross, labels)
        #pprint(tree_cross)
        accuracy = calculate_accuracy(pd.DataFrame(test_cross), tree_cross)
        percentile_accuracy = accuracy*100
        print("Cross Validation: Accuracy of block  %d is %.2f %%" %(i+1, percentile_accuracy))

    pr2_fig = pd.read_csv(filename, header=None)                              
    pr2_fig = np.array(pr2_fig, dtype = float)   
    plot_data(pr2_fig, 1)
    
