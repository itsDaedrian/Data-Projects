import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns

path_to_file = # removed for saftely reasons
df = pd.read_csv(path_to_file)
# print(df)

x = df[['temperature', 'humidity', 'windspeed']]
# x= df['humidity'].values.reshape(-1,1)
# x= df['windspeed'].values.reshape(-1,1)
y = df['rentals']
x.shape

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
SEED = 42
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = SEED)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# rentals = regressor.predict([[9.5]])

y_pred = regressor.predict(x_test)

df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

variables = ['temperature', 'humidity', 'windspeed']

for var in variables:
    plt.figure()

    sns.regplot(x=var, y='rentals', data=df).set(title=f'Regression plot of {var} and Rentals');

    plt.show()

feature_names = x.columns
model_coefficients = regressor.coef_

coefficients_df = pd.DataFrame(data = model_coefficients,
                              index = feature_names,
                              columns = ['Coefficient value'])
print(coefficients_df)