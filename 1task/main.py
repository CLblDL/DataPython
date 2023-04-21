import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# загрузка данных
builds_df = pd.read_csv("builds.csv", sep=",")
psus_df = pd.read_csv("psus.csv", sep=",")
builds_to_match_df = pd.read_csv("builds_to_match.csv", sep=",")

# объединяем builds_df и builds_to_match_df в один DataFrame
merged_df = pd.merge(builds_df, builds_to_match_df, on="Название сборки")

# объединяем merged_df и psus_df в один DataFrame
full_df = pd.merge(merged_df, psus_df, left_on="Блок питания", right_on="Модель")

# преобразование категориальных данных
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2, 4, 5, 9, 10, 11])], remainder='passthrough')
X = ct.fit_transform(full_df.iloc[:, :-1])
y = full_df.iloc[:, -1]

# разделение на тренировочный и тестовый наборы данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# обучение модели
model = LinearRegression().fit(X_train, y_train)

# оценка модели
print("R^2 на тренировочном наборе данных:", model.score(X_train, y_train))
print("R^2 на тестовом наборе данных:", model.score(X_test, y_test))