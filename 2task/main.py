import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Загружаем файл со средними показателями бойцов
fighters = pd.read_csv('fighters.csv')
# Загружаем файл с показателями боев
fights = pd.read_csv('fights.csv')

# Оставляем только необходимые колонки в датафрейме fighters:
fighters = fighters[['Name', 'Style', 'Reach', 'Striking Accuracy', 'Grappling Accuracy', 'Age']]

# Разделяем данные на обучающую и тестовую выборки:
X = fights.drop('Result', axis=1)
y = fights['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Обучаем модель на обучающей выборке:
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
# Предсказываем результаты для тестовой выборки и оцениваем точность модели:
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)

