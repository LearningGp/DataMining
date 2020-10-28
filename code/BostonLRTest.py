from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn.linear_model import LinearRegression

boston = load_boston()
X = boston.data
y = boston.target
df = pd.DataFrame(X, columns = boston.feature_names)
# print(df.head())
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.intercept_)
coeff_df = pd.DataFrame(regressor.coef_, df.columns, columns=['Coefficient'])
print(coeff_df)

y_pred = regressor.predict(X_test)
test_df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted':y_pred.flatten()})
print(test_df)

test_df1 = test_df.head(25)
test_df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print('MeanAbsolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('MeanSquared Error:', metrics.mean_squared_error(y_test, y_pred))
print('RootMean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
