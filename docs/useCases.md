# pycol: Use Cases

## Use Case I: Noise Removal

One practical use case of pycol is noise removal, particularly following synthetic data generation techniques like SMOTE
(Synthetic Minority Oversampling Technique). SMOTE often generates new instances randomly, which can result in synthetic
points in regions with high class overlap, introducing noise. By leveraging pycol’s overlap measures, we can identify
and remove these noisy synthetic instances. A scatter plot before and after noise removal can be obtained to see the results of the process (Figure xx).


![alt text](https://github.com/DiogoApostolo/pycol/blob/main/docs/images/SMOTE-1.png?raw=true){: width='40%' height='40%'}
![alt text](https://github.com/DiogoApostolo/pycol/blob/main/docs/images/SMOTE-2.png?raw=true){: width='40%' height='40%'}
![alt text](https://github.com/DiogoApostolo/pycol/blob/main/docs/images/SMOTE-3.png?raw=true){: width='40%' height='40%'}
![alt text](https://github.com/DiogoApostolo/pycol/blob/main/docs/images/SMOTE-4.png?raw=true){: width='40%' height='40%'}

Figure 1: Example of Noise Removal. The Minority Class is represented in Blue and the Majority Class is represented in Red. New minority samples (Dark Blue) are generated and removed according to their degree of overlap.

### Code Example

A pratical example using pycol is shown using the breast cancer dataset.

```python





from complexity import Complexity
from imblearn.over_sampling import SMOTE

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



```


## Use Case II: Guided Oversampling

Another valuable application is guided oversampling. Instead of applying a uniform oversampling strategy, we can use pycol to perform a more detailed oversampling based on the typology of safe, borderline, rare, and outlier instances, using the borderline metric. Specifically, pycol can be used to identify only one type of sample, for example the borderline samples, and use only these to generate new samples, instead of the entire dataset (Figure 2).


![alt text](https://github.com/DiogoApostolo/pycol/blob/main/docs/images/guide-1.png?raw=true){: width='30%' height='30%'}
![alt text](https://github.com/DiogoApostolo/pycol/blob/main/docs/images/guide-2.png?raw=true){: width='30%' height='30%'}
![alt text](https://github.com/DiogoApostolo/pycol/blob/main/docs/images/guide-3.png?raw=true){: width='30%' height='30%'}

Figure 2: Example of Guided Oversampling. The Minority Class is represented in Blue and the Majority Class is represented in Red. After the samples near the decision boundary are found, the dataset is oversampled using only these samples.



### Code Example

A practical use case is shown using the winequality dataset from the KEEL repository. In this example we are interested in dividing the samples of the dataset into safe, borderline, rare and outlier. The following code example displays how to obtain this division with pycol by using the borderline complexity measure with the return_all parameter set to True:


```python

comp = Complexity(file_name="datasets/winequality.arff")

B,S,R,O,C = comp.borderline(return_all=True,imb=True)

print(S)
print(B)
print(R)
print(O)

```

| Sample Type | Percentage Maj. Class | Percentage Min. Class |
| ------------- | ------------- | ------------- |
| Safe | 0.9827 | 0.000 |
| Borderline | 0.0173 | 0.0943 |
| Rare |  0.0000 | 0.2075 |
| Outlier |  0.000 | 0.6982 |

From these results, it is possible to observe that, for the minority class, there are many samples in the Outlier Class,
specifically about 70%. Oversampling this dataset uniformly would create many samples in the overlapped area, due to the
outlier samples. Performing oversampling just for borderline and rare samples in this case would be more beneficial, as it
would give visibility to the minority class without drastically increasing the overlapped area.

## Use Case III: Feature Selection

Feature selection is critical for building efficient and interpretable models. Pycol can assist in this process by evaluating and ranking features based on their discriminative power using the feature metrics such as F1 or F1v. Using pycol’s complexity measures, we can assess each feature’s contribution to class separability and select the most relevant ones.

A practical example is shown using an imbalanced credit card fraud detection dataset, from the OpenML repository. This dataset contains 30 features, and it is likely many of them can be removed without significantly losing classification performance. 

The goal is to pick the most discriminant features using the F1 overlap measure. Figure 3 shows all features plotted according to their discriminant power.

![alt text](https://github.com/DiogoApostolo/pycol/blob/main/docs/images/FeatureSelection.png?raw=true)

Figure 3: Discriminant Power of the Features of the credit card fraud Dataset

As it is possible to observe that there are some features with low discriminant power (below 0.6), which can likely be removed from the dataset without losing too much performance.


### Code Example

Using pycol we can do this by following this example:

```python


f1_average = 'binary'

folder = "datasets/"
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

```


| Dataset | Number of Features | Performance | 
| ------------- | ------------- | ------------- |
| Original Dataset | 30 | 0.8833 |
| After Preprocessing | 21  |0.8633 |

The results show that it's possible to use the F1 overlap measure for pre-processing, removing 9 features from the dataset without losing too much classification performance.


## Use Case IV: Algorithm Selection

Algorithm selection is a crucial step in the machine learning pipeline, where choosing the most suitable classification or preprocessing algorithm can significantly impact model performance. However, the effectiveness of an algorithm often depends on the underlying characteristics of the dataset, such as class imbalance and overlap. 

Pycol’s complexity measures can provide valuable insights into these characteristics, aiding in a more informed selection of an algorithm, through the use of meta learning. Particularly, by analysing a dataset with pycol, users can quantify aspects like class separability, feature interdependence, and sample density distribution. These insights help in predicting how different algorithms might perform.

In the following practical example, we show how complexity metrics can be used to choose the most adequate preprocessing algorithm. The goal is to show how structural complexity, particularly, ONB, can be used as an indicator for the choice of preprocessing algorithm.

To do this, we use two groups of datasets, one with low structural complexity (ONB lower than 0.3) and one with very high structural complexity (ONB higher than 0.7). For both groups of datasets, several pre-processing techniques are applied, which can be divided into three groups. The first group composed of three oversampling algorithms, the most popular oversampling technique, SMOTE and two of its most popular variants Borderline SMOTE and SMOTE-ENN. Group two, contains two popular undersampling techniques, Random Undersampling (RUS) and Repeated Edited Nearest Neighbours (REEN). Finally, group three contains two oversampling techniques that take into account the structural properties of the dataset: Graph SMOTE and MWMOTE.

Following this, the F-Measure of a kNN classifier is calculated on the original dataset and on the dataset after preprocessing, and the difference between these two values is calculated. 


### Code Example

```python


def oversample(method,classifier,X_train,y_train,X_test,y_test,f1_average):
        sm = method() 
        X_res, y_res = sm.fit_resample(X_train, y_train)
        classifier.fit(X_res, y_res)
        y_prob = classifier.predict_proba(X_test)
        y_pred = (y_prob[:,1] > 0.5).astype(int)        
        f1 = f1_score(y_test, y_pred, zero_division=0,average=f1_average,pos_label=1)
        return f1


#Load files with the datasets
folder = "datasets/"
onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
onlyfiles.sort(reverse=True)


#Measure the complexity
onb_dic = {}
dataset_dic = {}
for file in onlyfiles:
    complexity = Complexity(folder+file,distance_func="default",file_type="arff")
    onb_val = complexity.ONB(imb=True)[1]
    onb_dic[file] = onb_val
    dataset_dic[file] = [complexity.X, complexity.y]


df = pd.DataFrame(onb_dic.items(), columns=['dataset', 'ONB'])

#Select Classifier
f1_average = 'binary'
knn = KNeighborsClassifier(n_neighbors=5)


#Select the values with high and low complexity
difs_avg = []
good_bad_vals_df = df[(df['ONB']>0.7) | (df['ONB']<0.3)]


#Balance the datasets - run N times
max_versions = 10
for dataset in good_bad_vals_df['dataset']:
    
    smote_dif = []
    rus_dif = []
    reen_dif = []
    enn_dif = []
    border_dif = []
    gp_dif = []
    mw_dif = []
    
    ONB_val = good_bad_vals_df[good_bad_vals_df['dataset']==dataset]['ONB'].iloc[0]

    for version in range(max_versions):
        
        X = dataset_dic[dataset][0]
        y = dataset_dic[dataset][1]
        
        X_train, X_test, y_train, y_test = train_test_split(X,y , random_state=104, test_size=0.25,shuffle=True,stratify=y) 
        

        knn.fit(X_train, y_train)
        y_prob = knn.predict_proba(X_test)
        y_pred = (y_prob[:,1] > 0.5).astype(int)        
        f1_orig = f1_score(y_test, y_pred, zero_division=0,average=f1_average,pos_label=1)

        f1_smote = oversample(SMOTE,knn,X_train,y_train,X_test,y_test,f1_average)
        smote_dif.append(f1_smote - f1_orig)

        f1_rus = oversample(RandomUnderSampler,knn,X_train,y_train,X_test,y_test,f1_average)
        rus_dif.append(f1_rus - f1_orig)
        
        f1_reen = oversample(RepeatedEditedNearestNeighbours,knn,X_train,y_train,X_test,y_test,f1_average)
        reen_dif.append(f1_reen - f1_orig)

        f1_smoteenn = oversample(SMOTEENN,knn,X_train,y_train,X_test,y_test,f1_average)
        enn_dif.append(f1_smoteenn - f1_orig)
    
        f1_smoteborder = oversample(BorderlineSMOTE,knn,X_train,y_train,X_test,y_test,f1_average)
        border_dif.append(f1_smoteborder - f1_orig)

        f1_graph = oversample(smote_variants.SL_graph_SMOTE,knn,X_train,y_train,X_test,y_test,f1_average)
        gp_dif.append(f1_graph - f1_orig)




        f1_mw = oversample(smote_variants.MWMOTE,knn,X_train,y_train,X_test,y_test,f1_average)
        mw_dif.append(f1_mw - f1_orig)

    #Average the value for every run
    difs_avg.append([ONB_val,np.mean(smote_dif),np.mean(rus_dif),np.mean(reen_dif),np.mean(enn_dif),np.mean(border_dif),np.mean(gp_dif),np.mean(mw_dif)])



difs_df = pd.DataFrame(np.array(difs_avg),columns=['ONB','SMOTE','RUS','REEN','EEN','Borderline','GRAPH','MWMOTE'])

#Divide low and high complexity datasets
low_metric_df = difs_df[difs_df['ONB']<0.3]
high_metric_df = difs_df[difs_df['ONB']>0.7]

print(low_metric_df[['SMOTE','RUS','REEN','EEN','Borderline','GRAPH','MWMOTE']].mean())
print(high_metric_df[['SMOTE','RUS','REEN','EEN','Borderline','GRAPH','MWMOTE']].mean())



```

| Oversampling algorithm | SMOTE | SMOTE-ENN | Borderline | RUS | REEN | Graph | MWMOTE |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |     
| ONB < 0.3 | -0.009 | -0.0260 | -0.006  | -0.1648 | -0.0493 | -0.0310 |  0.0090 |
| ONB > 0.7 | 0.1002 | 0.1280  |  0.1245 |   0.0692| 0.1744  |0.1432   | 0.1572  |

From the results the following conclusions are presented:

1. For the datasets of low complexity, preprocessing does not show any improvement, on the contrary, in some cases it even shows a significant decrease;

2. The datasets with high structural complexity, almost always benefit from pre-processing, however undersampling techniques such as RENN or oversampling techniques that take into account the structural properties of the dataset like Graph SMOTE and MWMOTE, tend to perform the best.

This type of analysis can be done for other measures of the structural family, which if coupled with measures from other families can offer an even more complete picture of the dataset characteristics and aid in the choice of both preprocessing and classification algorithms.



## Use Case V: Performance Benchmark

Performance benchmarking involves comparing the effectiveness of different models or algorithms on a given dataset. Pycol can enhance this process by providing a detailed understanding of the dataset’s complexity, allowing for more nuanced benchmarking. When benchmarking models, it’s essential to consider not just the raw performance metrics (like accuracy, precision, recall) but also how these models interact with the inherent complexities of the dataset. Pycol enables this deeper analysis.

This repository presents a csv file (Benchmark.csv) with complexity measurements for 85 binary datasets. In particular, this file showcases several datasets with different types of overlap complexity, with each measure containing two values when possible, one for each class (Example: the measure N3 will have N3_1 for the first class and N3_2 for the second). Based on these measurements, a user can choose the type of classifier that's more appropriate for each dataset. For example, datasets with low local complexity will yield better results with neighbourhood based classifiers like kNN.