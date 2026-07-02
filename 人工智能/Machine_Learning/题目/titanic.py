from sklearn.datasets import fetch_openml
import pandas as pd

titanic = fetch_openml('titanic', version=1, as_frame=True)
df = titanic.frame

print(df.head())
print(df.info())
