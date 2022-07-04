"""
list comprehension to find low cardinality
"""

# list comprehension to find low cardinality
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]

# list comprehension to find numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# list comprehension to find high cardinality
high_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() > 10 and X_train_full[cname].dtype == "object"]

# list comprehension to find numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]