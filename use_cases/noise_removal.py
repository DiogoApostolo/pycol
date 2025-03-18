from complexity import Complexity
from imblearn.over_sampling import SMOTE, ADASYN
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


## Load Dataset##
n_samples_class1 = 150  # Number of points for the first class (center cluster)
n_samples_class2 = [20,5]  # Number of points for the second class (two smaller clusters, 100 each)
n_features = 2          # Number of features (2D for visualization)
cluster_std = 0.5       # Standard deviation for the clusters
random_state = 42       # Random seed for reproducibility


X1, y1 = make_blobs(n_samples=n_samples_class1,n_features=n_features,centers=[(0,0)],cluster_std=cluster_std,random_state=random_state)


X2, y2 = make_blobs(n_samples=n_samples_class2,n_features=n_features,centers=[(-3, 0), (1, 0)],cluster_std=cluster_std,random_state=random_state)

X = np.vstack((X1, X2))  # Stack the data points
y = np.hstack((np.zeros(n_samples_class1), np.ones(sum(n_samples_class2))))

plt.plot(X1[:, 0], X1[:, 1], 'o', label='Class 1')
plt.plot(X2[:, 0], X2[:, 1], 'o', label='Class 2')
plt.show()

#Oversample
sm = SMOTE(k_neighbors=7) 
X_res, y_res = sm.fit_resample(X, y)

dic = {'X':X_res, 'y':y_res}

plt.plot(X_res[y_res==0, 0], X_res[y_res==1, 1], 'o', label='Class 1')
plt.plot(X_res[y_res==1, 0], X_res[y_res==1, 1], 'o', label='Class 2')
plt.show()



#Measure Complexity
complexity = Complexity(dataset=dic,distance_func="default",file_type="array")
complexity_values = complexity.N3(inst_level=True,k=5)

#Remove noise: Find indexes of the new samples
original_samples_count = len(X)
new_samples_indexes = np.arange(original_samples_count, len(X_res))

#Remove noise: Find indexes of the sample with low complexity
low_complexity_inds = np.where(complexity_values<0.5)[0]

#Remove noise: Find the intersection of the two sets
intersect_array = np.intersect1d(low_complexity_inds, new_samples_indexes)

#Join original dataset with new samples 
union_array = np.union1d(np.arange(0,len(X)), intersect_array)
X_noise_removed = X_res[union_array,:]
y_noise_removed = y_res[union_array]

print("% Reduction: " + str(1 - len(X_noise_removed[y_noise_removed==1])/len(X_res[y_res==1])))

plt.plot(X_noise_removed[y_noise_removed==0, 0], X_noise_removed[y_noise_removed==0, 1], 'o', label='Class 1')
plt.plot(X_noise_removed[y_noise_removed==1, 0], X_noise_removed[y_noise_removed==1, 1], 'o', label='Class 2')
plt.show()
