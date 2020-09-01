# https://scikit-learn.org/stable/modules/naive_bayes.html

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

## second ex
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from  sklearn.datasets import  make_blobs
from sklearn.naive_bayes import GaussianNB
X, y = make_blobs(150, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');

model = GaussianNB()
model.fit(X,y)

# random new data set label
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14,18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)

plt.scatter(X[:, 0],X[:, 1],c=y , s=50, cmap='RdBu')
lim = plt.axis()

#plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap="RdBu", alpha=0.1)
# plt.axis(lim)

plt.show()



