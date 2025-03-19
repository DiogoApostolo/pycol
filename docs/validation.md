# Validation Test


In order to confirm the validity of the implemented measures, several validation tests were conducted. In particular the validation of the implemented measures was divided into two groups. The first group was used for the measures already implemented in pymfe, whose results were compared to those given by the measures implemented in the pycol package. The remaining measures without an available implementation were tested using synthetic datasets.

The artificial datasets were created using the data generator introduced in https://github.com/sysmon37/datagenerator/tree/newsampling. With this generator, it is possible to create data clusters of variable size and topology. The generator divides the creation of samples in multiple regions and the number of regions, their size, shape and location can all be configured by the user. For each type of shape available, there is an algorithm that uniformly fills the area inside this region with safe and borderline samples. Afterwards, the area around the region is populated with rare and outlier examples.


## Group I Datasets

Results for the first group of measures were compared with datasets of the KEEL repository. The characteristics of these datasets are shown in following table. The datasets were chosen to have a varying number of instances (from 205 to 2200) and features (from 4 to 7), as well as binary and non-binary classification problems. For non-binary datasets, the OvO results are summarized using a mean.

|Name | #Samples | #Features | #Classes |
| ------------- | ------------- | ------------- | ------------- |     
| newthyroid   | 215  | 5   | 3  |
| ecoli        | 335  | 7   | 8  |
| balance      | 625  | 4   | 3  |
| titanic      |2200  | 4   | 2  |


The results of the validation of the first group can be found in table bellow. All measures except from F1 and N2 obtain the exact same result in both packages for every dataset, indicating the implementation is indeed valid. As for F1, the difference in results is due to a slight change in the implementation where the means of each feature is not normalized, justifying variations between both approaches. Finally, for N2 the differences are also very small between the two packages, which is likely due to the default distance metrics used in each one of them, which are slightly different in terms of normalization.


| Measure | pycol (newthyroid) | pymfe (newthyroid) | pycol (ecoli) | pymfe (ecoli)| pycol (balance) | pymfe (balance) | pycol (titanic) | pymfe (titanic) |
|---------|-------------------|--------------------|---------------|---------------|-----------------|-----------------|-----------------|------------------|
| **F1**  | **0.5429**        | **0.5124**         | **0.5447**    | **0.5677**    | **0.8342**      | **0.8306**      | **0.8370**      | **0.9030**       |
| F1v     | 0.0613            | 0.0613             | 0.1186        | 0.1186        | 0.2904         | 0.2904          | 0.4356          | 0.4356           |
| F2      | 0.0007            | 0.0007             | 0.000         | 0.0000        | 1.000           | 1.0000          | 1.000           | 1.000            |
| F3      | 0.2216            | 0.2216             | 1.0           | 1.000         | 0.5980          | 0.5980          | 1.000           | 1.000            |
| F4      | 0.0               | 0.0                | 0.0636        | 0.0636        | 1.0000          | 1.0000          | 1.000          | 1.000            |
| N1      | 0.1023            | 0.1023             | 0.3035        | 0.3035        | 0.2752          | 0.2752          | 0.3198          | 0.3198           |
| **N2**  | **0.2368**        | **0.2478**         | **0.4160**    | **0.3966**    | **0.4036**      | **0.4231**      | **0.0270**      | **0**            |
| N3      | 0.0279            | 0.0279             | 0.2083        | 0.2083        | 0.2128          | 0.2128          | 0.2221          | 0.2221           |
| N4      | 0.0093            | 0.0093             | 0.1398        | 0.1398        | 0.1312          | 0.1312          | 0.4329          | 0.4329           |
| LSC     | 0.7702            | 0.7702             | 0.9741        | 0.9741        | 0.9663          | 0.9663          | 0.9999          | 0.9999           |
| T1      | 0.2279            | 0.2279             | 0.7529        | 0.7529        | 0.3648          | 0.3648          | 0.004           | 0.004            |


A code example of how to generate these values in pycol is also provided:


```python

from complexity import Complexity

folder = "dataset/group_one/"
onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]


for file in onlyfiles:
    complexity = Complexity(folder+file,distance_func="default",file_type="arff")
    f1_val = complexity.F1()
    f1v_val = complexity.F1v()
    f2_val = complexity.F2()
    f3_val = complexity.F3()
    f4_val = complexity.F4()
    n1_val = complexity.N1()
    n2_val = complexity.N2()
    n3_val = complexity.N3()
    n4_val = complexity.N4()
    lsc_val = complexity.LSC()
    t1_val = complexity.T1()

    print(f1_val)
    print(f1v_val)
    print(f2_val)
    print(f3_val)
    print(f4_val)

    print(n1_val)
    print(n2_val)
    print(n3_val)
    print(n4_val)

    print(lsc_val)
    print(t1_val)
```


## Group II Datasets

For the second group of measures, two sets of tests were made. The first set of tests starts by creating two clusters of different classes with 750 samples each. The overlapped region between these clusters is increased until the two clusters are completely overlapped. 

![alt text](https://github.com/DiogoApostolo/pycol/blob/new_main/docs/images/no_overlap.png?raw=true){: width='40%' height='40%'}
![alt text](https://github.com/DiogoApostolo/pycol/blob/new_main/docs/images/some_overlap.png?raw=true){: width='40%' height='40%'}
![alt text](https://github.com/DiogoApostolo/pycol/blob/new_main/docs/images/lot_overlap.png?raw=true){: width='40%' height='40%'}
![alt text](https://github.com/DiogoApostolo/pycol/blob/new_main/docs/images/full_overlap.png?raw=true){: width='40%' height='40%'}



Ideally, if the complexity measures are implemented correctly, their values will indicate a higher complexity as the overlapped region increases. Results for the second group of metrics in each of the artificial datasets are presented in the table bellow. 


| Measure       | Test 1   | Test 2   | Test 3   | Test 4   |
|---------------|----------|----------|----------|----------|
| **R value**   | 0.003    | 0.1140   | 0.2953   | 0.7107   |
| **D3**        | 2.5      | 85.5  | 221.5 | 533|
| **CM**        | 0.003    | 0.114    | 0.2953   | 0.7106   |
| **kDN**       | 0.0052   | 0.0957   | 0.2406   | 0.58413  |
| **DBC**       | 0.7142   | 0.9672   |  0.9781   | 0.96268   |
| **SI**        | 0.9966   | 0.888   | 0.658   | 0.3533   |
| **input noise**| 0.4990  | 0.5927   | 0.7410   | 0.9983   |
| **borderline**| 0.008   | 0.1113  | 0.3326  | 0.758  |
| **deg overlap**| 0.0147  | 0.1753   | 0.4346   | 0.6182   |
| **C1**        | 0.0245   | 0.1204   | 0.2908   | 0.61036   |
| **C2**        | 0.004   | 0.1021   | 0.2751   | 0.5031   |
| **Clst**      | 0.004    | 0.1220   | 0.3366   | 0.7147   |
| **purity**    | 0.0843   |  0.0846   | 0.0880   | 0.2108   |
| **neigh. sep.**|  0.0292  | 0.0279  | 0.0256   | 0.1228   |


Overall, the results show that all the metrics behave according to the expected, as when the overlapped region increases, their values increase too. A notable exception to this rule are the SI, purity and neighbourhood separability measures, however these measures work differently from the rest where smaller values indicate higher complexity, so the values presented still indicate that the implementation is valid. 


Afterwards, a second set of more complex artificial datasets was used, which were taken from https://github.com/sysmon37/datagenerator/tree/newsampling. The following table  presents the characteristics of the datasets and a 2D view of the datasets is presented as well. 



| **Name**     | **#Samples** | **#Features** | **#Classes** | **Class Ratio** |
|--------------|--------------|---------------|--------------|-----------------|
| circles-2d   | 800          | 2             | 2            | 1:3             |
| spheres-2d   | 3000         | 2             | 2            | 1:1             |
| paw3-2d      | 1000         | 2             | 2            | 1:9             |
| paw3-3d      | 1500         | 2             | 3            | 1:7             |





![alt text](https://github.com/DiogoApostolo/pycol/blob/new_main/docs/images/circles2d.png?raw=true){: width='40%' height='40%'}
![alt text](https://github.com/DiogoApostolo/pycol/blob/new_main/docs/images/spheres2d.png?raw=true){: width='40%' height='40%'}
![alt text](https://github.com/DiogoApostolo/pycol/blob/new_main/docs/images/paw2d.png?raw=true){: width='40%' height='40%'}
![alt text](https://github.com/DiogoApostolo/pycol/blob/new_main/docs/images/paw3d.png?raw=true){: width='40%' height='40%'}





As most of the experimented metrics take into account the local region around each sample, it is expected that the values for the measures will represent lower complexity, since as seen in the figures these datasets present low local overlap, with very well-defined clusters. An exception is input noise, which is a feature based metric and should have high values, since none of the two features is able to linearly separate any of the datasets. The results of these experiments are presented bellow and are within expectations, as most of the measures indicate a low complexity. The measures between 0 and 1 are lower than 0.5, when 1 represents high complexity and higher than 0.5 when 1 represents low complexity.  Also, as expected, input noise, being a feature based metric, gives very high values, representing high complexity. The two measures that got results which defied expectations were purity and neighbourhood separability, which both have similar formulations. However, being multi-resolution metrics, this result is most likely due to the need for better parametrization, which is very dataset dependent.


| Measure       | Circles-2D   | sphere-2d   | paw3-3d   | paw3-2d   |
|---------------|-----------|-----------|-----------|-----------|
| **R value**   | 0.4783    | 0.01967   | 0.2296    | 0.6222    |
| **D3**        | 0.82      | 0.295       | 0.354       | 0.4   |
| **CM**        | 0.205     | 0.01967   | 0.118     | 0.08      |
| **kDN**       | 0.2557    | 0.0334    | 0.1664    | 0.1366    |
| **DBC**       | 0.9531    | 0.6575    | 0.9610    | 0.9743    |
| **SI**        | 0.7325   | 0.9783    | 0.8333    | 0.848     |
| **input noise**| 0.9568   | 0.7958    | 0.9886    | 0.9760    |
| **borderline**| 0.20375    | 0.0536     | 0.1153   | 0.111   |
| **deg overlap**| 0.5787   | 0.0923    | 0.394    | 0.3600    |
| **C1**        | 0.2782   | 0.0435     | 0.2276       | 0.1645    |
| **C2**        | 0.2597    | 0.0282    | 0.1667    | 0.1412    |
| **Clst**      | 0.2987    | 0.02467   |  0.17    | 0.1590    |
| **purity**    | 0.0716     | 0.1034   | 0.0695     | 0.0689     |
| **neigh. sep.**| 0.0215   | 0.0288    | 0.0185    | 0.0277    |


The values for both experiments can be obtained by running the code below:


```python

from complexity import Complexity

folder = "dataset/group_two_part_one/"  ## "dataset/group_two_part_two/" change to get the results from the first or second table
onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
onlyfiles.sort(reverse=True)

onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

for file in onlyfiles:

    print(file)

    complexity = Complexity(folder+file,distance_func="default",file_type="arff")
    
    

    R_val = complexity.R_value()
    R_val = sum(R_val)/len(R_val)

    d3_val = complexity.D3_value()
    d3_val = sum(d3_val)/len(d3_val) /100
    cm_val = complexity.CM()
    kdn_val = complexity.kDN()
    dbc_val = complexity.DBC()
    si_val = complexity.SI()
    in_val = complexity.input_noise()
    in_val = sum(in_val)/len(in_val)

    borderline_val = complexity.borderline()
    deg_val = complexity.deg_overlap()
    C1_val = complexity.C2()
    C2_val = complexity.C1()
    clust_val = complexity.Clust()
    pur_val = complexity.purity()
    neigh_val = complexity.neighbourhood_separability()

    print("R measure: ",R_val)
    print("D3 measure: ",d3_val)  
    print("CM measure: ",cm_val)
    print("kDN measure: ",kdn_val)
    print("DBC measure: ",dbc_val)
    print("SI measure: ",si_val)
    print("input noise measure: ",in_val)
    print("borderline measure: ",borderline_val)
    print("degOver measure: ",deg_val)
    print("C1 measure: ",C1_val)
    print("C2 measure: ",C2_val)
    print("Clust measure: ",clust_val)
    print("purity measure: ",pur_val)
    print("neighbourhood separability measure: ",neigh_val)
```


