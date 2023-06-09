import pandas as pd
import tarfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import StratifiedShuffleSplit


def load_data():
    project_path = Path.cwd()
    tarball_path = project_path / "housing.tgz"
    
    if tarball_path.is_file():
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
        return pd.read_csv(Path("datasets/housing/housing.csv"))
    else:
        print("The files are missing")

housing = load_data()      

"""
       
# Description of the data

housing.info() 


# describe shows a numeric value of numerical attributes 

print(housing.describe())

# Deeper look at ocean_proximity
print(housing["total_bedrooms"].value_counts())



# Histogram for attributes
housing.hist(bins=50,figsize=(12,8))
plt.show()



# # Creating a test set using a random generator and seed
# np.random.seed(42)

# def shuffle_and_split_data(data,test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]

# train_set,test_set = shuffle_and_split_data(housing,0.2)
    
# print(len(train_set))
# print(len(test_set))


# A more scalable way to split the data into test and training set
def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32  


def split_data_with_id(data, test_ratio, columnId):
    
    ids = data[columnId]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_,test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index() # adding index that we can use in the function above

train_set, test_set = split_data_with_id(housing_with_id, 0.2, "index")


"""

# Splitting the data with an equal income category 
housing["income_cat"] = pd.cut(housing["median_income"],bins=[0.,1.5,3.0,4.5,6,np.inf], labels=[1,2,3,4,5])

# Stratified object
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []

for train_index, test_index in splitter.split(housing,housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n,strat_test_set_n])
    
strat_train_set, strat_test_set = strat_splits[0]


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
 
housing = strat_train_set.copy()

# # Visual representation of high/low populated areas
# housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, s = housing["population"]/100, 
#              label = "population", c = "median_house_value", cmap = "jet", colorbar =True, legend = True,sharex = False, figsize = (10,7))
# plt.show()

# Standard correlation between attributes
corr_matrix = housing.corr()

corr_median_house_value = corr_matrix["median_house_value"].sort_values(ascending= False)
print(corr_median_house_value)

housing.plot(kind = "scatter", x = "median_income",y = "median_house_value", alpha = 0.1, grid = True)
plt.show()
