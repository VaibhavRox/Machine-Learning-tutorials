import pandas as pd
import math

# --- Calculate entropy of a column ---
def entropy(column):
    values = column.unique()
    total = len(column)
    ent = 0
    for val in values:
        p = len(column[column == val]) / total
        ent += -p * math.log2(p)
    return ent

# --- Information gain for a feature ---
def info_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values = data[feature].unique()
    weighted_entropy = 0
    
    for val in values:
        subset = data[data[feature] == val]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target])
    
    return total_entropy - weighted_entropy

# --- ID3 recursive function ---
def id3(data, features, target):
    # Case 1: If all target values are same → return that value
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]
    
    # Case 2: If no features left → return majority class
    if len(features) == 0:
        return data[target].mode()[0]   # most frequent class
    
    # Find the best feature (highest information gain)
    best_feature = None
    best_gain = -1
    for f in features:
        gain = info_gain(data, f, target)
        if gain > best_gain:
            best_gain = gain
            best_feature = f
    
    # Build the tree as a dictionary
    tree = {best_feature: {}}
    
    # Split dataset by the best feature values
    for val in data[best_feature].unique():
        subset = data[data[best_feature] == val]
        
        # Remove used feature
        new_features = [f for f in features if f != best_feature]
        
        # Recursive call
        subtree = id3(subset, new_features, target)
        
        tree[best_feature][val] = subtree
    
    return tree

# --- Prediction function ---
def predict(tree, sample):
    # If tree is a leaf node (not dict), return the class
    if not isinstance(tree, dict):
        return tree
    
    # Otherwise, get the root feature
    feature = list(tree.keys())[0]
    value = sample.get(feature)
    
    # If value not in tree (unseen case), return None
    if value not in tree[feature]:
        return None
    
    # Recursive call
    return predict(tree[feature][value], sample)

# --- Main ---
if __name__ == "__main__":
    # Load CSV file
    filename = "data.csv"   # Change this to your CSV file name
    df = pd.read_csv(filename)

    target = df.columns[-1]        # assume last column is target
    features = list(df.columns[:-1])

    # Build tree
    decision_tree = id3(df, features, target)
    print("Decision Tree:")
    print(decision_tree)

    # --- Ask user for input ---
    print("\nEnter values to classify:")
    user_sample = {}
    for f in features:
        val = input(f"Enter {f}: ")
        user_sample[f] = val

    # Predict class
    result = predict(decision_tree, user_sample)
    print("\nPredicted class:", result)
