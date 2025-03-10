# pycol: Use Cases

## Use Case I: Noise Removal

One practical use case of pycol is noise removal, particularly following synthetic data generation techniques like SMOTE
(Synthetic Minority Oversampling Technique). SMOTE often generates new instances randomly, which can result in synthetic
points in regions with high class overlap, introducing noise. By leveraging pycol’s overlap measures, we can identify
and remove these noisy synthetic instances. A scatter plot before and after noise removal can be obtained to see the results of the process (Figure \ref{fig:noise-rem-1}).

![alt text](https://github.com/DiogoApostolo/pycol/blob/main/images/SMOTE-1.png?raw=true)
![alt text](https://github.com/DiogoApostolo/pycol/blob/main/images/SMOTE-2.png?raw=true)
![alt text](https://github.com/DiogoApostolo/pycol/blob/main/images/SMOTE-3.png?raw=true)
![alt text](https://github.com/DiogoApostolo/pycol/blob/main/images/SMOTE-4.png?raw=true)

### Code Example

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


*Figure 1: Example of Noise Removal. The Minority Class is represented in Blue and the Majority Class is represented in Red. New minority samples (Dark Blue) are generated and removed according to their degree of overlap.

## Use Case II: Guided Oversampling

Another valuable application is guided oversampling. Instead of applying a uniform oversampling strategy, we can use pycol to perform a more detailed oversampling based on the typology of safe, borderline, rare, and outlier instances \citep{borderline}, using the borderline metric. Specifically, pycol can be used to identify only one type of sample, for example the borderline samples, and use only these to generate new samples, instead of the entire dataset (Figure \ref{fig:noise-rem}).

![alt text](https://github.com/DiogoApostolo/pycol/blob/main/images/guide-1.png?raw=true)
![alt text](https://github.com/DiogoApostolo/pycol/blob/main/images/guide-2.png?raw=true)
![alt text](https://github.com/DiogoApostolo/pycol/blob/main/images/guide-3.png?raw=true)

A practical use case is shown using the \textit{winequality} dataset from the KEEL repository. In this example we are interested in dividing the samples of the dataset into safe, borderline, rare and outlier. The following code example displays how to obtain this division with pycol by using the borderline complexity measure with the \textit{return\_all} parameter set to True:


### Code Example

```python

comp = Complexity(file_name="winequality.arff")

B,S,R,O,C = comp.borderline(
    return_all=True,imb=True)

print(B)
print(S)
print(R)
print(O)

```

## Use Case III: Feature Selection

Feature selection is critical for building efficient and interpretable models. Pycol can assist in this process by evaluating and ranking features based on their discriminative power using the feature metrics such as F1 or F1v. Using pycol’s complexity measures, we can assess each feature’s contribution to class separability and select the most relevant ones.

A practical example is shown using an imbalanced credit card fraud detection dataset, from the OpenML repository. This dataset contains 30 features, and it is likely many of them can be removed without significantly losing classification performance. 

The goal is to pick the most discriminant features using the F1 overlap measure. Figure \ref{fig:feature_sel} shows all features plotted according to their discriminant power.

![alt text](https://github.com/DiogoApostolo/pycol/blob/main/images/Feature-selection-2.png?raw=true)

As it is possible to observe that there are some features with low discriminant power (below 0.6), which can likely be removed from the dataset without losing too much performance.

### Code Example

```python

param_grid = {'C': [1, 10],  
                'gamma': [0.1, 0.01], 
                'kernel': ['rbf']} 

f1_average = 'binary'

folder = "arff/"
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

The results show that it's possible to use the F1 overlap measure for pre-processing, removing 9 features from the dataset without losing too much classification performance.


## Use Case IV: Algorithm Selection

Algorithm selection is a crucial step in the machine learning pipeline, where choosing the most suitable classification or preprocessing algorithm can significantly impact model performance. However, the effectiveness of an algorithm often depends on the underlying characteristics of the dataset, such as class imbalance and overlap. 

Pycol’s complexity measures can provide valuable insights into these characteristics, aiding in a more informed selection of an algorithm, through the use of meta learning. Particularly, by analysing a dataset with pycol, users can quantify aspects like class separability, feature interdependence, and sample density distribution. These insights help in predicting how different algorithms might perform.

In the following practical example, we show how complexity metrics can be used to choose the most adequate preprocessing algorithm. The goal is to show how structural complexity, particularly, ONB, can be used as an indicator for the choice of preprocessing algorithm.

To do this, we use two groups of datasets, one with low structural complexity (ONB lower than 0.3) and one with very high structural complexity (ONB higher than 0.7). For both groups of datasets, several pre-processing techniques are applied, which can be divided into three groups. The first group composed of three oversampling algorithms, the most popular oversampling technique, SMOTE and two of its most popular variants Borderline SMOTE and SMOTE-ENN. Group two, contains two popular undersampling techniques, Random Undersampling (RUS) and Repeated Edited Nearest Neighbours (REEN). Finally, group three contains two oversampling techniques that take into account the structural properties of the dataset: Graph SMOTE and MWMOTE.

Following this, the F-Measure of a kNN classifier is calculated on the original dataset and on the dataset after preprocessing, and the difference between these two values is calculated. The results on the two groups of datasets can be found in Table \ref{tab:oversample_test}, showing that:


1. For the datasets of low complexity, preprocessing does not show any improvement, on the contrary, in some cases it even shows a significant decrease;

2. The datasets with high structural complexity, almost always benefit from pre-processing, however undersampling techniques such as RENN or oversampling techniques that take into account the structural properties of the dataset like Graph SMOTE and MWMOTE, tend to perform the best.

This type of analysis can be done for other measures of the structural family, which if coupled with measures from other families can offer an even more complete picture of the dataset characteristics and aid in the choice of both preprocessing and classification algorithms.

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




## Use Case V: Performance Selection

Performance benchmarking involves comparing the effectiveness of different models or algorithms on a given dataset. Pycol can enhance this process by providing a detailed understanding of the dataset’s complexity, allowing for more nuanced benchmarking. When benchmarking models, it’s essential to consider not just the raw performance metrics (like accuracy, precision, recall) but also how these models interact with the inherent complexities of the dataset. Pycol enables this deeper analysis.