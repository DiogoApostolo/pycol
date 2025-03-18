from complexity import Complexity
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.datasets import load_breast_cancer

## Load Dataset##
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target


#Oversample
sm = SMOTE() 
X_res, y_res = sm.fit_resample(X, y)

dic = {'X':X_res, 'y':y_res}


#Measure Complexity
complexity = Complexity(dataset=dic,distance_func="default",file_type="array")
complexity_values = complexity.N3(inst_level=True,k=5)

#Remove noise: Find indexes of the new samples
original_samples_count = len(X)
new_samples_indexes = np.arange(original_samples_count, len(X_res))

#Remove noise: Find indexes of the sample with low complexity
low_complexity_inds = np.where(complexity_values<0.7)[0]

#Remove noise: Find the intersection of the two sets
intersect_array = np.intersect1d(low_complexity_inds, new_samples_indexes)

#Join original dataset with new samples 
union_array = np.union1d(np.arange(0,len(X)), intersect_array)
X_noise_removed = X_res[union_array,:]

print("% Reduction: " + str(1 - len(X_noise_removed)/len(X_res)))
