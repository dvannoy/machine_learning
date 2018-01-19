import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import seaborn

filename = "data/san_diego_redfin_data.tsv"
redfin_df = pd.read_csv(filename, delimiter='\t', header=0)

print(redfin_df.head())

X = redfin_df['Avg Sale To List'].values.reshape(-1,1) # added processing since a single feature
y = redfin_df['Sold Above List']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)


print(X_train)
print(y_train)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

print("Accuracy", model.score(X_test, y_test))

pred = model.predict(.92)
print(pred)

plt.scatter(X_test, y_test)
plt.plot(X_test, model.predict(X_test))
plt.title('Evaluate model with ...', fontsize=10)
plt.show()