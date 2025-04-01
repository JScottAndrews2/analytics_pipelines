import pandas as pd
# NOTE: unused import
import numpy as np
# NOTE: same package called twice
from factor_analyzer import FactorAnalyzer
from factor_analyzer import ConfirmatoryFactorAnalyzer, ModelSpecificationParser


data = pd.read_csv("data/satisfaction/survey (1).csv")

# NOTE:sloppy drop of last column, which is assumed to be the dependent variable
data = data.iloc[:, 0:-1]
features = ['A1a', 'A1b', 'A2a', 'A2b', 'A2c', 'A2d', 'A2e', 'A2f', 'A2g', 'A2h', 'A3a', 'A3b', 'A3c', 'A3d',
            'A3e', 'A3f', 'A3g', 'A3h', 'A3i', 'A3j', 'A3k', 'A3l', 'A3m', 'A4', 'A4ai', 'A4aii', 'A4aiii',
            'A4aiv', 'A4av', 'A4avi', 'A4avii', 'B1a', 'B1b', 'B2a', 'B2b', 'B2c', 'B2d', 'B3a', 'B3b', 'B3c',
            'B3d', 'B4a', 'B4b', 'B4c', 'B4d', 'B5a', 'B5b', 'B5c', 'B5d', 'B6', 'B7a', 'B7b', 'B7c', 'B7d',
            'B7e', 'B8', 'B8ai', 'B8aii', 'B8aiii', 'B8aiv', 'B8av', 'B8avi', 'B8avii', 'B9', 'C1a', 'C1b', 'C1c',
            'C1d', 'C1e', 'C1f', 'C1g', 'C2', 'C3', 'C4a', 'C4b', 'C4c', 'C4d', 'C4e', 'C4f', 'C4g', 'C4h', 'C4i',
            'C4j', 'C4k', 'C4l', 'c4m', 'C4n', 'C4o', 'C4ii', 'C5', 'C6', 'C6ai', 'C6aii', 'C6aiii', 'C6aiv',
            'C6av', 'C6avi', 'C6avii', 'C6aviii', 'C6aix', 'C6ax', 'C6axi', 'C7', 'C8', 'C9', 'D3', 'D8', 'D14']
dep_var = 'AgencySize'

na_counts = data.isna().sum()
# Keep only variables with missing counts lower than 100
keep_vars = [col for col in na_counts.index if na_counts[col] < 100]

# drop rows with missing values from the reduced variables dataset
clean_data = data[keep_vars].dropna().reset_index(drop=True)

# split data into train and test #
x = data[features]
y = data[dep_var]
# NOTE: import called within scripts instead of at the top
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,
                                                    random_state=123)

fa = FactorAnalyzer(rotation='varimax')
# NOTE: OOOPS, forgot to pass in the training data...
fa.fit(clean_data)
# Check Eigenvalues
ev = fa.get_eigenvalues()
print(ev)
n_factors = 5
fa = FactorAnalyzer(rotation='varimax', n_factors=n_factors)
fa.fit(clean_data.values)

loadings = fa.loadings_
loadings = pd.DataFrame(loadings)
loadings.index = clean_data.columns
factors = loadings.idxmax(axis=1)
factors.columns = ['factor_number']

# NOTE: this loop can be greatly condensed using a list comprehension
scale_dict = {}
for current_factor in range(n_factors):
    item_names = []
    for var in factors.iteritems():
        if var[1] == current_factor:
            item_names.append(var[0])
    scale_dict[current_factor] = item_names

# NOTE: what am I doing here? I really should leave a meaningful note!!
vars = [item for sublist in scale_dict.values() for item in sublist]
cfa_data = clean_data[vars]

model_spec = ModelSpecificationParser.parse_model_specification_from_dict(cfa_data, scale_dict)
cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)

cfa.fit(cfa_data.values)
# NOTE: did I want to randomly print this? Look for unused code and remove it.
cfa.loadings_
cfa.get_standard_errors()

# OK, things are already too messy for me and I stopped!
# There would be several hundred more lines of difficult to navigate code and I don't feel like making more of that for
# no real purpose.
#Let's talk about some straegies to help clean-up this code and also make is more dynamic and re-usable.
# Think about this, if you find that you always need to split your data,
# why not make a class to handle data for every project you work on?