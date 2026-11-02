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

# Step 7 - compare_trees
def compare_trees(X, y, max_depth=2):
    
    # fit both th trees
    my_tree = grow_tree(X,y,max_depth = max_depth , min_samples_leaf = 1)
    sk_tree = fit_sklearn_tree(X,y, max_depth = max_depth, min_samples_leaf = 1)

    # splits extraction from my tree (recursively)
    my_splits = []
    def collect_splits(node):
        if not node['leaf']:
            my_splits.append((node['feature'], round(node['threshold'],3)))
            collect_splits(node['left'])
            collect_splits(node['right'])
    
    collect_splits(my_tree)
    
    sk_splits  = sklearn_splits(sk_tree)

    my_preds = predict_tree(my_tree, X)
    sk_preds = sk_tree.predict(X)

    agreement = float(np.mean(my_preds == sk_preds))

    return{
        'same_splits' : my_splits == sk_splits,
        'agreement' : agreement,
        'my_splits' : my_splits
    }

# Step 8 - moons_data
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def moons_data(n_samples=300, noise=0.25, random_state=42, test_size=0.3):
    # TODO: make_moons + train_test_split; return (X_train, X_test, y_train, y_test).
    Points , labels = make_moons(n_samples, noise = noise, random_state = random_state)

    X_train,X_test,y_train,y_test = train_test_split(Points,labels,test_size = test_size, random_state = random_state)
    return (X_train,X_test,y_train,y_test)

# Step 9 - overfit_vs_regularized
def calculate_accuracy(y_true, y_pred): 
    return float(np.mean(y_true == y_pred))

def overfit_vs_regularized(X_train, X_test, y_train, y_test, min_samples_leaf=5):
    
    unrestricted_tree = fit_sklearn_tree(X_train, y_train, max_depth=None, min_samples_leaf=1)
    reg_tree = fit_sklearn_tree(X_train, y_train, max_depth=None, min_samples_leaf=min_samples_leaf)

    # predict on train and test sets
    y_pred_train_unrestricted = unrestricted_tree.predict(X_train)
    y_pred_test_unrestricted  = unrestricted_tree.predict(X_test)
    y_pred_train_reg = reg_tree.predict(X_train)
    y_pred_test_reg = reg_tree.predict(X_test)

    # calculate accuracies using helper function
    train_acc_unres = calculate_accuracy(y_train, y_pred_train_unrestricted)
    test_acc_unres = calculate_accuracy(y_test, y_pred_test_unrestricted)
    train_acc_reg = calculate_accuracy(y_train, y_pred_train_reg)
    test_acc_reg = calculate_accuracy(y_test, y_pred_test_reg)

    leaves_unres = int(unrestricted_tree.get_n_leaves())
    leaves_reg = int(reg_tree.get_n_leaves())

    return {
        'free': {
            'train_acc': train_acc_unres,
            'test_acc': test_acc_unres,
            'leaves': leaves_unres
        },
        'regularized': {
            'train_acc': train_acc_reg,
            'test_acc': test_acc_reg,
            'leaves': leaves_reg
        }
    }

# Step 10 - rotation_sensitivity
def rotation_sensitivity(X_train, X_test, y_train, y_test, degrees=45.0, min_samples_leaf=5):
    # convert degrees to radians
    theta = np.deg2rad(degrees)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ])

    # 2. Rotate both training and testing feature matrices
    X_train_rot = X_train @ R.T
    X_test_rot = X_test @ R.T

    # 3. Fit trees on original and rotated training data
    orig_tree = fit_sklearn_tree(X_train, y_train, max_depth=None, min_samples_leaf=min_samples_leaf)
    rot_tree = fit_sklearn_tree(X_train_rot, y_train, max_depth=None, min_samples_leaf=min_samples_leaf)

    # 4. Evaluate test accuracies
    orig_preds = orig_tree.predict(X_test)
    rot_preds = rot_tree.predict(X_test_rot)

    original_acc = float(np.mean(orig_preds == y_test))
    rotated_acc = float(np.mean(rot_preds == y_test))
    drop = round(original_acc - rotated_acc, 4)

    return {
        "original_acc": original_acc,
        "rotated_acc": rotated_acc,
        "drop": drop
    }

# Step 11 - regression_tree
from sklearn.tree import DecisionTreeRegressor

def regression_tree(X, y, max_depth=2, random_state=42):
    
    model = DecisionTreeRegressor(max_depth = max_depth , random_state = random_state)
    model.fit(X,y)

    y_pred = model.predict(X)

    # find unique prediction values 
    n_distinct_predictions = int(np.unique(y_pred).size)

    # training_loss
    train_mse = float(np.mean((y - y_pred)**2))

    return {
        'model' : model,
        'n_distinct_predictions': n_distinct_predictions,
        'train_mse' : train_mse
    }

# Step 12 - tree_rules
from sklearn.tree import export_text

def tree_rules(clf, feature_names):
    return export_text(clf, feature_names = list(feature_names))

# Step 13 - save_and_reload_tree (not yet solved)
# TODO: implement

# Step 14 - predict_species (not yet solved)
# TODO: implement

