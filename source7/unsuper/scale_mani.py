from sklearn.datasets import make_blobs
import matplotlib.pylab as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import mglearn

#sample data 생성 
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
#train data와 test data로 나눔 
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

#train data와 test data의 산점도
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(X_train[:, 0], X_train[:, 1],
                c=mglearn.cm2(0), label="train set", s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^',
                c=mglearn.cm2(1), label="test set", s=60)
axes[0].legend(loc='upper left')
axes[0].set_title("original data")

# 스케일조정
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 스케일이 조정된 데이터의 산점도
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                c=mglearn.cm2(0), label="train set", s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^',
                c=mglearn.cm2(1), label="test set", s=60)
axes[1].set_title("scaled data")
#plt.show()

# # 테스트 세트의 스케일을 따로 조정합니다
# # 테스트 세트의 최솟값은 0, 최댓값은 1이 됩니다
# # 이는 예제를 위한 것으로 절대로 이렇게 사용해서는 안됩니다
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)
 
# 잘못 조정된 데이터의 산점도를 그립니다
axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                c=mglearn.cm2(0), label="training set", s=60)
axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1],
                marker='^', c=mglearn.cm2(1), label="test set", s=60)
axes[2].set_title("mistaken data")
 
for ax in axes:
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
fig.tight_layout()
plt.show()