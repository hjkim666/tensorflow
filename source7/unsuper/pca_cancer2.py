from sklearn.preprocessing.data import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mglearn
cancer = load_breast_cancer()

scalar = StandardScaler()
scalar.fit(cancer.data)
X_scaled = scalar.transform(cancer.data)

pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
print("original data: {}".format(str(X_scaled.shape)))
print("reduction data: {}".format(str(X_pca.shape)))

plt.figure(figsize=(8,8))
mglearn.discrete_scatter(X_pca[:,0], X_pca[:,1], cancer.target)
plt.legend(["malignancy(cancer)", "benign"], loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("first principal component")
plt.ylabel("second  principal component")
plt.show()


print("PCA PC shape: {}".format(pca.components_.shape))
print("PCA PC: {}".format(pca.components_))
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ["first principal component", "second  principal component"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),
           cancer.feature_names, rotation=60, ha='left')
plt.xlabel("feature")
plt.ylabel("principal component")
plt.show()
