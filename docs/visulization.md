# Visualization Options

## Complexity Bar Plot

To facilitate an easy understanding of all the dimensions of overlap, pycol allows users to easily display the values of each overlap family. Pycol stores each measure previously calculated, so when the visualization method is called, all measures calculated so far are displayed. Particularly, each family is displayed using a different colour in one bar plot.

![alt text](https://github.com/DiogoApostolo/pycol/blob/new_main/docs/images/viz_metrics.png?raw=true)

An example of how to generate this visualization in displayed below:  


```python

from complexity import Complexity

complexity = Complexity(file_name="dataset/winequality_red_4.arff")

'''Calculate Instance measures'''
complexity.kDN()
complexity.N3()
complexity.N4()

'''Calculate Structural measures'''
complexity.N1()
complexity.N2()


'''Calculate Feature measures'''
complexity.F1()
complexity.F1v()

'''Calculate Multi-Resolution measures'''
complexity.C1()
complexity.C2()

''' Display the bar plots '''
complexity.viz_metrics()


```

## PCA Projection

A user might want to visualize the difficulty of each instance in the dataset, instead of a single averaged
measure of the entire dataset. Pycol accommodates this feature by calculating the instance hardness of each sample and
then computing a Principal Component Analysis (PCA) of the dataset, reducing it to the two most relevant components.
The data is displayed in two dimensions, while simultaneously using a colour gradient to indicate the hardness of
each sample. It’s worth mentioning that calculating instance hardness is based on the neighbourhood of each instance,
so a k value for the number of nearest neighbours must be chosen (in the case of the example k=11).

A pratical example is shown for two datasets: the iris dataset and the glass dataset. The iris dataset, which is known to be easy to classify, displays very low instance hardness for both classes. Additionally, the glass dataset shows two clusters with low complexity, which represent regions dominated by one class, and a cluster in the middle where samples from both classes are present, resulting in higher instance hardness.


```python

from complexity import Complexity

complexity = Complexity(file_name="dataset/61_iris_binary.arff")

```

![alt text](https://github.com/DiogoApostolo/pycol/blob/new_main/docs/images/iris_PCA.png?raw=true)


```python

from complexity import Complexity

complexity = Complexity(file_name="dataset/alg_sel/glass5.arff")

```

![alt text](https://github.com/DiogoApostolo/pycol/blob/new_main/docs/images/glass_PCA.png?raw=true)




