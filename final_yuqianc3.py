import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import collections


def load_file():
    if not os.path.exists("countyComplete.csv"):
        print("File not exist")
    dfs = pd.read_csv("countyComplete.csv")
    dfs = pd.DataFrame.dropna(dfs, 1)
    total_person = dfs['households'] * dfs['persons_per_household']
    dfs['total_person'] = total_person
    dfs.to_csv("new_county.csv")
    return dfs
# The above code will load the csv data, remove the na value and add a new line to the csv, finally export a new csv file.

dfs = load_file()

x = np.array(dfs['hs_grad']).reshape(-1, 1)
y = np.array(dfs['poverty'])
lm_profile = linear_model.LinearRegression()
lm_profile.fit(X=x, y=y)
a, b = lm_profile.coef_, lm_profile.intercept_
print("The slope is: " + str(a) + ", the intercept is " + str(b))

plt.scatter(x, y, color='blue', s=1)
plt.plot(x, lm_profile.predict(X=x), color='red', linewidth=4)
plt.show()
# The above code will do a linear regression between the variable hs_grad and poverty, then draw a plot for the regression line.


interval = 2000
income_counter = collections.Counter()
for ele in dfs['per_capita_income']:
    income_counter[ele // interval] += 1

income = [i for i in income_counter.keys()]
income_amounts = [i for i in income_counter.values()]
plt.bar(income, income_amounts, color='blue')
plt.xlabel('Per Capital Income (k)')
plt.ylabel('Amount')
plt.show()

# The above code will draw the per capita income distribution histogram. Take every 2000 as an interval.
