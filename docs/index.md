[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)

# pycol: Python Class Overlap Library

The Python Class Overlap Library (`pycol`) assembles a set of data complexity measures associated to the problem of class overlap. 

The combination of class imbalance and overlap is currently one of the most challenging issues in machine learning. However, the identification and characterisation of class overlap in imbalanced domains is a subject that still troubles researchers in the field as, to this point, there is no clear, standard, well-formulated definition and measurement of class overlap for real-world domains.

This library characterises the problem of class overlap according to multiple sources of complexity, where four main class overlap representations are acknowledged: **Feature Overlap**, **Instance Overlap**, **Structural Overlap**, and **Multiresolution Overlap**.

Existing open-source implementations of complexity measures include the [DCoL](https://github.com/nmacia/dcol) (C++), [ECoL](https://github.com/lpfgarcia/ECoL), and the recent [ImbCoL](https://github.com/victorhb/ImbCoL), [SCoL](https://github.com/lpfgarcia/SCoL), and [mfe](https://github.com/rivolli/mfe) packages (R code). There is also 
[pymfe](https://github.com/ealcobaca/pymfe) in Python. Regarding class overlap measures, these packages consider the implementation of the following: F1, F1v, F2, F3, F4, N1, N2, N3, N4, T1 and LSCAvg. `ImbCoL` further provides a decomposition by class of the original measures and `SCoL` focuses on simulated complexity measures. In order to foster the study of a more comprehensive set of measures of class overlap, we provide an extended Python library, comprising the class overlap measures included in the previous packages, as well as an additional set of measures proposed in recent years. Furthermore, this library implements additional adaptations of complexity measures to class imbalance. 

Overall, `pycol` characterises class overlap as a heterogeneous concept, comprising distinct sources of complexity, and the following measures are implemented:


#### Feature Overlap:
* **F1:** Maximum Fisher's Discriminat Ratio
* **F1v:** Directional Vector Maximum Fisher's Discriminat Ratio
* **F2:** Volume of Overlapping Region
* **F3:** Maximum Individual Feature Efficiency
* **F4:** Collective Feature Efficiency
* **IN:** Input Noise


#### Instance Overlap:
* **R-value**
* **Raug:** Augmented R-value
* **degOver**
* **N3:** Error Rate of the Nearest Neighbour Classifier
* **SI:** Separability Index
* **N4:** Non-Linearity of the Nearest Neighbour Classifier
* **kDN:** K-Disagreeing Neighbours
* **D3:** Class Density in the Overlap Region
* **CM:** Complexity Metric Based on k-nearest neighbours
* **wCM:** Weighted Complexity Metric
* **dwCM:** Dual Weighted Complexity Metric
* **Borderline Examples**
* **IPoints:** Number of Invasive Points


#### Structural Overlap:
* **N1:** Fraction of Borderline Points
* **T1:** Fraction of Hyperspheres Covering Data
* **Clst:** Number of Clusters
* **ONB:** Overlap Number of Balls
* **LSCAvg:** Local Set Average Cardinality
* **DBC:** Decision Boundary Complexity
* **N2:** Ratio of Intra/Extra Class Nearest Neighbour Distance
* **NSG:** Number of samples per group
* **ICSV:** Inter-class scale variation


#### Multiresolution Overlap:
* **MRCA:** Multiresolution Complexity Analysis
* **C1:** Case Base Complexity Profile
* **C2:** Similarity-Weighted Case Base Complexity Profile
* **Purity**
* **Neighbourhood Separability**

## Instalation

All packages required to run pycol are listed in the requirements.txt file. 
To install all needed pacakges run:

`pip install -r requirements.txt`

The package is also available for instalation through pypi: https://pypi.org/project/pycol-complexity/

## Usage Example:

The `dataset` folder contains some datasets with binary and multi-class problems. All datasets are numerical and have no missing values. The `complexity.py` module implements the complexity measures.
To run the measures, the `Complexity` class is instantiated and the results may be obtained as follows:

```python
from complexity import Complexity
complexity = Complexity("dataset/61_iris.arff",distance_func="default",file_type="arff")

# Feature Overlap
print(complexity.F1())
print(complexity.F1v())
print(complexity.F2())
# (...)

# Instance Overlap
print(complexity.R_value())
print(complexity.deg_overlap())
print(complexity.CM())
# (...)

# Structural Overlap
print(complexity.N1())
print(complexity.T1())
print(complexity.Clust())
# (...)

# Multiresolution Overlap
print(complexity.MRCA())
print(complexity.C1())
print(complexity.purity())
# (...)
```

## Developer notes:
To submit bugs and feature requests, report at [project issues](https://github.com/DiogoApostolo/pycol/issues).

## Licence:
The project is licensed under the MIT License - see the [License](https://github.com/DiogoApostolo/pycol/blob/main/LICENCE) file for details.

## Acknowledgements:
Some complexity measures implemented on `pycol` are based on the implementation of `pymfe`. We also thank José Daniel Pascual-Triana for providing the implementation of ONB.
