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

# Step 3 - grow_tree (not yet solved)
# TODO: implement

# Step 4 - predict_tree (not yet solved)
# TODO: implement

# Step 5 - fit_sklearn_tree (not yet solved)
# TODO: implement

# Step 6 - sklearn_splits (not yet solved)
# TODO: implement

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

