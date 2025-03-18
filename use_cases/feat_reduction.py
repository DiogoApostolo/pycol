from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from complexity import Complexity
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt


f1_average = 'binary'

folder = "../dataset/"
file = "creditCardFraud.arff"

#Chose Classifier
classifier = xgb.XGBClassifier()
threshold = 0.60

#Measure Feature Complexity
complexity = Complexity(file_name=folder+file,distance_func="default",file_type="arff")
f1_list = complexity.F1()

my_dict = {f'F{i+1}': value for i, value in enumerate(f1_list)}


#Show Bar Plot
plt.bar(range(len(my_dict)), list(my_dict.values()), align='center')
plt.xticks(range(len(my_dict)), list(my_dict.keys()))
plt.show()



#Make Classification With the Full Dataset
X = complexity.X
y = complexity.y

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25,shuffle=True,stratify=y) 
            

classifier.fit(X_train, y_train)
y_prob = classifier.predict_proba(X_test)
y_pred = (y_prob[:,1] > 0.5).astype(int)        
f1_orig = f1_score(y_test, y_pred, zero_division=0,average=f1_average,pos_label=1)


#Choose Features above the threshold
features = np.where(np.array(f1_list) > threshold)[0]


#Keep Only the relevant Features
X_train_reduced = X_train[:,features]
X_test_reduced = X_test[:,features]


#Make Classification with the Reduced Dataset
classifier.fit(X_train_reduced, y_train)
y_prob = classifier.predict_proba(X_test_reduced)
y_pred = (y_prob[:,1] > 0.5).astype(int)        
f1_reduced = f1_score(y_test, y_pred, zero_division=0,average=f1_average,pos_label=1)


print(f1_orig)
print(f1_reduced)
