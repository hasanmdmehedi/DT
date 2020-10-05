import numpy as np
import pandas as pd

eps = np.finfo(float).eps
from numpy import log2 as log
import pprint

df = pd.read_csv("data/synthetic-1.csv", header=None, skiprows=0, )

##1. claculate entropy o the whole dataset
entropy_node = 0  # Initialize Entropy
values = df[2].unique()  # Unique objects - 'Yes', 'No'

for value in values:
    fraction = df[2].value_counts()[value] / len(df[2])
    entropy_node += -fraction * np.log2(fraction)


# Now define a function {ent} to calculate entropy of each attribute :
def ent(df, attribute):
    target_variables = df[2].unique()  # This gives all 'Yes' and 'No'
    variables = df[attribute].unique()  # This gives different features in that attribute (like 'Sweet')

    entropy_attribute = 0
    for variable in variables:
        entropy_each_feature = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute] == variable][df[2] == target_variable])  # numerator
            den = len(df[attribute][df[attribute] == variable])  # denominator
            fraction = num / (den + eps)  # pi
            entropy_each_feature += -fraction * log(
                fraction + eps)  # This calculates entropy for one feature like 'Sweet'
        fraction2 = den / len(df)
        entropy_attribute += -fraction2 * entropy_each_feature  # Sums up all the entropy ETaste

    return (abs(entropy_attribute))


# store entropy of each attribute with its name :

a_entropy = {k: ent(df, k) for k in df.keys()[:-1]}


# calculate Info gain of each attribute :
def ig(e_dataset, e_attr):
    return (e_dataset - e_attr)


# store IG of each attr in a dict :
# entropy_node = entropy of dataset
# a_entropy[k] = entropy of k(th) attr
IG = {k: ig(entropy_node, a_entropy[k]) for k in a_entropy}

def find_entropy(df):
    Class = df.keys()[-1]  # To make the code generic, changing target variable class name
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value] / len(df[Class])
        entropy += -fraction * np.log2(fraction)
    return entropy


def find_entropy_attribute(df, attribute):
    Class = df.keys()[-1]  # To make the code generic, changing target variable class name
    target_variables = df[Class].unique()  # This gives all 'Yes' and 'No'
    variables = df[
        attribute].unique()  # This gives different features in that attribute (like 'Hot','Cold' in Temperature)
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute] == variable][df[Class] == target_variable])
            den = len(df[attribute][df[attribute] == variable])
            fraction = num / (den + eps)
            entropy += -fraction * log(fraction + eps)
        fraction2 = den / len(df)
        entropy2 += -fraction2 * entropy
    return abs(entropy2)


def find_winner(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
        #         Entropy_att.append(find_entropy_attribute(df,key))
        IG.append(find_entropy(df) - find_entropy_attribute(df, key))
    return df.keys()[:-1][np.argmax(IG)]


def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)


def buildTree(df, tree=None):
    Class = df.keys()[-1]  # To make the code generic, changing target variable class name

    # Here we build our decision tree

    # Get attribute with maximum information gain
    node = find_winner(df)

    # Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
    attValue = np.unique(df[node])

    # Create an empty dictionary to create tree
    if tree is None:
        tree = {}
        tree[node] = {}

    # We make loop to construct a tree by calling this function recursively.
    # In this we check if the subset is pure and stops if it is pure.

    for value in attValue:

        subtable = get_subtable(df, node, value)
        clValue, counts = np.unique(subtable[Class], return_counts=True)

        if len(counts) == 1:  # Checking purity of subset
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = buildTree(subtable)  # Calling the function recursively

    return tree


main_tree = buildTree(df)
pprint.pprint(main_tree)

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

#from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 2
plot_colors = "ryb"
plot_step = 0.02

# Load data
#iris = load_iris()

for pairidx, pair in enumerate([[0, 1], [0, 2]]):
    # We only take the two corresponding features
    X = df[[0,1]].to_numpy()
    y = df[2].to_numpy()

    # Shuffle
    idx = np.arange(X.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    # Train
    clf = DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary
    #plt.subplot(1, 2, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    plt.axis("tight")

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color,cmap=plt.cm.Paired)

    plt.axis("tight")

plt.suptitle("synthetic-1")
plt.legend()
plt.show()
