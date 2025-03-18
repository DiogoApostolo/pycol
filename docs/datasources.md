# pycol: Data Sources

## Artificial Datasets
To explore the complexity implemented in *pycol*, the user may refer to the `dataset` folder [in the GitHub repository](https://github.com/DiogoApostolo/pycol). 

Alternatively, it is also possible to generate custom artificial datasets using the a data generator that outputs files in `.arff` format ([you may find available documentation here](https://github.com/miriamspsantos/datagenerator)).

In case the user wishes to select datasets with specific complexity characteristics, the pycol package also offer an extensive benchmark of previously computed complexity measures, [available in this .csv file](https://github.com/DiogoApostolo/pycol/blob/new_main/Benchmark.csv). The used datasets for this benchmark are also available in the `dataset/alg_sel` folder [here](https://github.com/DiogoApostolo/pycol/tree/new_main/dataset/alg_sel).


## Benchmark of Imbalanced Datasets
To experiment with a large benchmark of imbalanced datasets, the user is referred to [KEEL Datastet Repository](http://www.keel.es), containing a selection of several datasets categorized by IR.


## Other Real-World Datasets
There are several publicly available data sources that can be used while exploring *pycol*:

- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Irvine Machine Learning Repository](https://archive.ics.uci.edu)
- [OpenML](https://www.openml.org)


## Reading datasets into pycol

The first step when using pycol is to instantiate the `Complexity` class. When doing this, the user must provide  the dataset that is going to be analysed, the distance function that is going to be used to calculate the distance between samples and the file type. The example below showcases the analysis of the dataset `61_iris.arff`, choosing the default distance function (HEOM) and specifying that the dataset is in the arff format:

```python
Complexity('61_iris.arff',
           distance_func='default',
           file_type='arff')
```

Alternatively, a user might want to load a dataset directly into pycol from an array, for example after fetching a dataset from `sklearn`. To do this, the user must specify the `file_type` argument as "array" and provide a python dictionary with the keys `X`, containing the data and `y` containing the target labels.


```python    
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
dic = {'X':X, 'y':y}
complexity = Complexity(dataset=dic,
             file_type="array")
```