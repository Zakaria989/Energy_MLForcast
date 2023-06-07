import pandas as pd
import tarfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


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
"""
# Deeper look at ocean_proximity
print(housing["total_bedrooms"].value_counts())


# # # Select the variables you want to plot
# # variables_to_plot = ['temp', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed']

# # # Create subplots for each variable
# # fig, axes = plt.subplots(len(variables_to_plot), 1, figsize=(12, 8))

# # # Iterate over each variable and plot its histogram
# # for i, variable in enumerate(variables_to_plot):
# #     axes[i].hist(weather[variable], bins=50)
# #     axes[i].set_xlabel(variable)
# #     axes[i].set_ylabel('Frequency')

# # plt.tight_layout()
# # plt.show()



# # Creating a test set
# np.random.seed(42)

# def shuffle_and_split_data(data,test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]

# data = weather.drop_duplicates()

# train_set,test_set = shuffle_and_split_data(weather,0.2)



# print(len(train_set))
# print(len(test_set))
    