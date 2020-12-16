import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

df = pd.read_csv('Airplane_Crashes_and_Fatalities_Since_1908.csv')
df.info()

df = df.drop('Flight #',axis=1)
df = df.drop('Time',axis=1)
df = df.drop('Route',axis=1)
df = df.drop('cn/In',axis=1)
df = df.drop('Summary',axis=1)
df.notnull()

X = df.drop('Location',axis=1)
X = X.drop('Type',axis=1)
X = X.drop('Date',axis=1)
X = X.drop('Operator',axis=1)
X = X.drop('Ground',axis=1)
X = X.drop('Registration',axis=1)
X = X.dropna()
xc=X

X = X.values
y = df['Type'].values

samples = X.astype(int)

scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
model = make_pipeline(scaler, kmeans)
model.fit(samples)

labels = model.predict(samples)
xs = samples[:, 0]
ys = samples[:, 1]
plt.scatter(xs, ys, c=labels)

centroids = kmeans.cluster_centers_
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]
plt.scatter(centroids_x, centroids_y, color='Red', marker='D', s=50)

plt.show()

centroids = kmeans.cluster_centers_
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]
plt.scatter(centroids_x, centroids_y, marker='D', s=50)

plt.show()

fig, ax = plt.subplots()
ax.bar(xs, ys, width=6, color='r')
plt.show()
