"""
Example 1-1 shows the Python code that loads the data,
separates the inputs X from the labels y, creates a scatterplot for visualization, and
then trains a linear model and makes a prediction.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Download and p(repare the data
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat.csv")
X = lifesat[["GDP per capita(USD)"]].values
y = lifesat[["Life satisfaction"]].values

# Visualize the data
lifesat.plot(kind="scatter", grid=True, x="GDP per capita(USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# Select a linear model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[37_655.2]]                # Cyprus' GDP per capita in 2020
print(model.predict(X_new))         # output: [[6.30165767]]
