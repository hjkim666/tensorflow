from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
import mglearn
from sklearn.decomposition import PCA


people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])
plt.show()   

print(people.target[0:10], people.target_names[people.target[0:10]])

print("people.image.shape: {}".format(people.images.shape))
print("클래스 개수: {}".format(len(people.target_names)))   

###################################
counts = np.bincount(people.target) 

for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='  ')
    if (i + 1) %3 ==0:
        print() 
mask = np.zeros(people.target.shape, dtype=np.bool)

for target in np.unique(people.target):
    mask[np.where(people.target==target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

X_people = X_people/255.        
###################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 데이터를 훈련 세트와 테스트 세트로 나눕니다
X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0)
# 이웃 개수를 한 개로 하여 KNeighborsClassifier 모델을 만듭니다
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("\n1-최근접 이웃의 테스트 세트 점수: {:.2f}".format(knn.score(X_test, y_test)))

###################################
# import mglearn
# mglearn.plots.plot_pca_whitening() 
# pca = PCA(n_components=100, whiten=True,random_state=0)
# pca.fix(X_train)
# X_train_pca = pca.transform(X_train)
#X_test_pca = pca.transform(X_test)




