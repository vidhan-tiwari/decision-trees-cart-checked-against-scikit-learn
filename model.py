"""
Decision Trees: CART, Checked Against Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - gini
def gini(y):
    # base case is needed if array is empty counts becomes 0 makes 
    # gini to be 1.0 which is incorrect 
    if y.size == 0 :
        return 0.0

    _ , counts = np.unique(y,return_counts = True)
    partial_fraction = counts/y.size
    gini = 1.0 - np.sum(partial_fraction**2)
    return float(gini)

# Step 2 - best_split
def best_split(X, y, min_samples_leaf = 1):
    # adding the min_sample_leaf parameter as we need not to consider the split that 
    # result in child with n_samples lower than min_sample_leaf no matter how pure that split becomes 
   
    n_samples , n_features = X.shape
    parent_gini = gini(y)

    best_f = None
    best_t = None
    best_gain = -1.0

    # for every feature/column :
    for f in range(n_features):
        # calculate how many unique values in that column
        # this return sorted unique values in asc order  
        unique_val, count = np.unique(X[:,f],return_counts = True)
        if count.size == 1:
            return(None, None, 0.0)
        # candidate thresholds are mid points of two 
        # consecutive sorted values 
        # numpy way of calculating sum of consecutive numbers  
        thresholds = (unique_val[:-1] + unique_val[1:])/2.0
        # size of thresholds is exact 1 less than the size of unique vals

        # for every threshold calculate left and right split
        for t in thresholds:
            # following masks are boolena Values
            t = float(t)
            left_mask = X[:,f] <= t
            right_mask = ~ left_mask

            # only take corresponding label where mask is TRUE
            y_left = y[left_mask]
            y_right = y[right_mask]

            # if this split results into a child having sample less than min_samples_leaf ignore the split 
            if len(y_left) < min_samples_leaf or len(y_right) < min_samples_leaf:
                continue
                
            # calulate weighted gini of splits 
            w_gini_left = (len(y_left)/n_samples)*gini(y_left)
            w_gini_right = (len(y_right)/n_samples)*gini(y_right)

            # calculate gain 
            gain = parent_gini -(w_gini_left + w_gini_right)

            # store if best 
            if gain > best_gain:
                best_gain = float(gain)
                best_f = f 
                best_t = t 
    return best_f, best_t , best_gain

# Step 3 - grow_tree
def grow_tree(X, y, max_depth=2, min_samples_leaf=1, depth=0):
    n_samples, n_features = X.shape
    
    classes, counts = np.unique(y, return_counts = True)

    # find out the majority class 
    majority_class = int(classes[np.argmax(counts)])

    # a tempplate for leaf_node 
    leaf_node = {'leaf' : True, 'value': majority_class, 'n' : n_samples}

    # if parent is pure i.e. paren_gini is 0.0 or the max_depth has been achieved return the leaf node
    parent_gini = gini(y)

    if parent_gini == 0.0 or depth >= max_depth:
        return leaf_node

    # now apply the best_split function.
    best_f, best_t , best_gain = best_split(X, y, min_samples_leaf = min_samples_leaf)

    # if gain was not updated(i.e. best_f , best_t remained None) return the current as leaf node 
    if best_t == None: # indicated no improvement in gain 
        return leaf_node
    
    # prepare left_mask and right_mask for best_t , best_f on X , y to make recursion
    left_mask = X[:,best_f] <= best_t
    right_mask = ~ left_mask

    # recursive calls for left and right subtrees
    left_subtree = grow_tree(X[left_mask], y[left_mask], max_depth , min_samples_leaf , depth + 1)
    right_subtree = grow_tree(X[right_mask], y[right_mask], max_depth , min_samples_leaf, depth + 1)

    return {
        'leaf' : False,
        'feature' : best_f,
        'threshold' : best_t,
        'n' : n_samples,
        'left' : left_subtree,
        'right' : right_subtree
    }

# Step 4 - predict_tree
def predict_tree(tree, X):
    # TODO: walk the dict tree per row; return an int array of leaf values.
    # a typical tree structure is : 
    '''
    {
    'leaf' : False,
    'feature' : best_f,
    'threshold' : best_t,
    'n' : n_samples,
    'left' : left_subtree,
    'right' : right_subtree
    }
    what to do with 'n' at prediction time?
    '''
    # loop - wise approach. 
    # initialise an array of zeros that would be my answer 
    labels = np.zeros(X.shape[0], dtype = int)
    for i in range(X.shape[0]):
        curr_node = tree
        while not curr_node['leaf']:
            f = curr_node['feature']
            t = curr_node['threshold']
            if X[i,f] <= t:
                curr_node = curr_node['left']
            else:
                curr_node = curr_node['right']
        labels[i] = curr_node['value']
    return labels

# Step 5 - fit_sklearn_tree
from sklearn.tree import DecisionTreeClassifier
def fit_sklearn_tree(X, y, max_depth=2, min_samples_leaf=1, random_state=42):
    model =  DecisionTreeClassifier(max_depth = max_depth , min_samples_leaf = min_samples_leaf , random_state = random_state)
    model.fit(X,y)
    return model

# Step 6 - sklearn_splits
def sklearn_splits(clf):
    tree = clf.tree_
    splits = []
    
    # Iterate through all nodes in node-index order
    for i in range(tree.node_count):
        # Skip leaves (where children_left is -1)
        if tree.children_left[i] != -1:
            feature = int(tree.feature[i])
            threshold = round(float(tree.threshold[i]), 3)
            splits.append((feature, threshold))
            
    return splits

# Step 7 - compare_trees (not yet solved)
# TODO: implement

# Step 8 - moons_data (not yet solved)
# TODO: implement

# Step 9 - overfit_vs_regularized (not yet solved)
# TODO: implement

# Step 10 - rotation_sensitivity (not yet solved)
# TODO: implement

# Step 11 - regression_tree (not yet solved)
# TODO: implement

# Step 12 - tree_rules (not yet solved)
# TODO: implement

# Step 13 - save_and_reload_tree (not yet solved)
# TODO: implement

# Step 14 - predict_species (not yet solved)
# TODO: implement

