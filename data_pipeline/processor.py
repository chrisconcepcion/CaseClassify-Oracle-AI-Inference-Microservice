# Importing pandas as we want to use vectorized operations which
# are 100x times faster than python loops. 
# We also want to use other data manipulation libraries like numpy, 
# scikit-learn, etc. 
import pandas as pd 

# Pipeline bundles multiple cleaning steps into a single obkect. 
# This is useful for reproducibility and makes it easy to apply the
# the same cleaning steps to multiple datasets.
from sklearn.pipeline import Pipeline 

# Now we need something to deal with None and NaN values. 
# SimpleImputer "fills the gaps".
from sklearn.impute import SimpleImputer 

# We also need to standardize the text data since the model expects
# numbers to keep things simple.
# - StandardScaler: Scales numbers to a common scale, a mean of 0 and a standard deviation of 1.
# - OneHotEncoder: Converts categories to numbers.
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Data going through cleaning is only required on specific columns.
# For exmple we don't want to clean a field that's clean like a numerical value field
# excluding the case_id.
from sklearn.compose import ColumnTransformer

# Use numpy for impossible values like NaN and None, -1, etc..
import numpy as np 

# 1. Define our "Contracts" (Which columns get which treatment)
NUMERIC_FEATURES = ['case_age_days', 'claim_amount']
CATEGORICAL_FEATURES = ['case_type', 'jurisdiction']

def get_reproducible_pipeline():
    """
    This function returns a single object that remembers how to clean data.
    """

    # A. Numeric Recipe
    # If data is missing use the Median. Then scale it so the
    # model doesn't get confused by different units(days vs dollars).
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # B. Categorical Recipe
    # If a category is missing or simply unknown then one
    # hot encode it into 9's and 1's to the math works.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # C. Column Transformer
    # Apply different treatments to different columns.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor 



def ensure_data_integrity(df: pd.DataFrame):
    """
    Proactivately protecting the model from poisoned data.
    """

    # 1. Defensive Filtering:
    # If a claim amount is negative or exceeds a logical threshold it's more than likely an error.
    # We drop it. 
    df = df[df['claim_amount'] > 0].copy() 

    # 2. Logical Validation:
    # A case can't have previous_appealrs if the case_age_days is 0.
    # We can use np.where to fix contradictions like this. 
    df['previous_appeals'] = np.where(df['case_age_days'] < 0, 0, df['previous_appeals'])

    return df 

def add_business_features(df: pd.DataFrame):
    """
    Turning raw numbers into domain-specific signals
    """

    # 1. Binary Flags
    # High-value cases often require different legal scrutiny. 
    df['is_high_value'] = (df['claim_amount'] > 25000).astype(int)
    
    # 2. Derived Ratios
    # Cost per day might be a better predictor of urgency than just cost.
    df['cost_intensity'] = df['claim_amount'] / (df['case_age_days'] + 0.001)

    return df 

def clean_data_efficiently(df: pd.DataFrame):
    """
    Vectorized operations > Python loops.
    """
    
    # 1. Vectorized Filter
    # Remove negative claim amounts by creating a mask and applying it.
    df = df[df['claim_amount'] >= 0].copy()

    # 2. Vectorized Transformation
    # Flag cases over 25k as high value
    df['is_high_value'] = np.where(df['claim_amount'] > 25000, 1, 0)

    # 3. Clean up whitespace 
    df['jurisdiction'] = df['jurisdiction'].str.strip().str.upper()
    
    return df 

    