import copy
import math
from operator import itemgetter
import numpy as np
from numpy.core.fromnumeric import shape, transpose, var
import arff
from sklearn.cluster import KMeans
import sklearn
import sklearn.pipeline
import scipy.spatial
import pandas as pd











class Complexity:
    '''
    Complexity class, it makes available the following methods to calculate complexity metrics:
    F1, F1v, F2, F3, F4, R_value, D3, CM, kDN, MRCA, C1, C2, T1, DBC, N1, N2, N3, N4, SI,
    LSC, purity, neighbourhood_seperability, input_noise, borderline, deg_overlap, ICSV, NSG, Clust, L1, L2, L3

    '''

    def __init__(self,file_name,distance_func="default",file_type="arff"):
        '''
        Constructor method, setups up the the necessary class attributes to be
        used by the complexity measure functions.
        Starts by reading the file in arff format which contains the class samples X (self.X), class labels y (self.y) and contextual information
        about the features (self.meta).
        It then calculates the distance matrix that contains the distance between all points in X (self.dist_matrix).
        It also saves in an array the unique labels of all existing classes (self.classes), the number of samples in each class (self.class_count) and
        the indexes in X of every class (self.class_inxs).
        -----
        Parameters:
        file_name (string): Location of the file that contains the dataset.
        distance_func (string): The distance function to be used to calculate the distance matrix. Only available option right now is "HEOM".
        file_type (string): The type of file where the dataset is stored. Only available option right now is "arff".
        
        '''
        if(file_type=="arff"):
            [X,y,meta]=self.__read_file(file_name)
        else:
            print("Only arff files are available for now")
            return

        self.X=np.array(X)
        #print(self.X)
        #self.X = self.X[:,0:2]
        self.y=np.array(y)
        classes=np.unique(self.y)

        self.classes = classes
        self.meta=meta
        self.dist_matrix,self.unnorm_dist_matrix = self.__calculate_distance_matrix(self.X,distance_func=distance_func)
        self.class_count = self.__count_class_instances()


        self.class_inxs = self.__get_class_inxs()

        if(len(self.class_count)<2):
           print("ERROR: Less than two classes are in the dataset.")

        return 


    
    
    def __count_class_instances(self):
        '''
        Is called by the __init__ method.
        Count instances of each class in the dataset.
        --------
        Returns:
        class_count (numpy.array): An (Nx1) array with the number of intances for each of the N classes in the dataset 
        '''
        class_count = np.zeros(len(self.classes))
        for i in range(len(self.classes)):
            count=len(np.where(self.y == self.classes[i])[0])
            class_count[i]+=count
        return class_count

    
    
    
    def __read_file(self,file_name):
        '''
        Is called by the __init__ method.
        Read an arff file containing the dataset.
        --------
        Parameters:
        file_name: string, name of the arff file to read from
        --------
        Returns:
        X (numpy.array): An array containing the attributes of all samples;
        y (numpy.array): An array containing the class labels of all samples; 
        meta (array): An array with information about the type of attributes (numerical or categorical) 
        --------
        '''

        f = arff.load(open(file_name, 'r'))
        data = f['data']
        num_attr = len(data[0])-1
        

    
        att=f['attributes']
        meta=[]

        #for every attribute check if it is numeric or categorical
        for i in range(len(att)-1):
            if(att[i][1]=="NUMERIC"):
                meta.append(0)
            else:
                meta.append(1)


        



        # print(meta)
        #split each sample into attributes and label 
        X = [i[:num_attr] for i in data]
        y = [i[-1] for i in data]

        #print(X)

        #convert categorical features to ordinal
        X = np.array(X)
        for i in range(len(meta)):
            if meta[i]==1:
                
                b, c = np.unique(X[:,i], return_inverse=True)
                X[:,i] = c

        if 1 in meta:
            X = X.astype(np.float)


        #indentify the existing classes
        classes = np.unique(y)

        #create new class labels from 0 to n, where n is the number of classes
        y = [np.where(classes == i[-1])[0][0] for i in data]
        

        #print(X)        

        return [X,y,meta]
    
    
    def __distance_HEOM(self,X):
        '''
        Is called by the calculate_distance_matrix method.
        Calculates the distance matrix between all pairs of points from an input matrix, using the HEOM metric, that way categorical attributes are
        allow in the dataset.
        --------
        Parameters: 
        X (numpy.array): An (N*M) numpy matrix containing the points, where N is the number of points and M is the number of attributes per point.
        --------
        Returns:
        dist_matrix (numpy.array): A (M*M) matrix containing the distance between all pairs of points in X
        '''
        
        meta= self.meta
        
        dist_matrix=np.zeros((len(X),len(X)))
        unnorm_dist_matrix = np.zeros((len(X),len(X)))

        #calculate the ranges of all attributes
        range_max=np.max(X,axis=0)
        range_min=np.min(X,axis=0)
        #print(range_max)
        #print(range_min)
        for i in range(len(X)): 
            for j in range(i+1,len(X)):
                #for attribute
                dist = 0
                unnorm_dist = 0
                for k in range(len(X[0])):
                    #missing value
                    if(X[i][k] == None or X[j][k]==None):
                        dist+=1
                        unnorm_dist+=1
                    #numerical
                    if(meta[k]==0):
                        #dist+=(abs(X[i][k]-X[j][k]))**2
                        
                        #dist+=(abs(X[i][k]-X[j][k])/(range_max[k]-range_min[k]))**2
                        if(range_max[k]==range_min[k]):
                            dist+=(abs(X[i][k]-X[j][k]))**2
                            unnorm_dist+=(abs(X[i][k]-X[j][k]))**2
                        else:
                            dist+=(abs(X[i][k]-X[j][k])/(range_max[k]-range_min[k]))**2
                            unnorm_dist+= abs(X[i][k]-X[j][k])**2
                            
                            #dist+=(abs(X[i][k]-X[j][k]))**2
                    #categorical
                    if(meta[k]==1):
                        if(X[i][k]!=X[j][k]):
                            dist+=1
                            unnorm_dist+=1

                dist_matrix[i][j]=np.sqrt(dist)
                dist_matrix[j][i]=np.sqrt(dist)

                unnorm_dist_matrix[i][j]=np.sqrt(unnorm_dist)
                unnorm_dist_matrix[j][i]=np.sqrt(unnorm_dist)
        #print(dist_matrix)
        return dist_matrix,unnorm_dist_matrix

    def __distance_HEOM_different_arrays(self,X,X2):
        '''
        Calculates the distance matrix between all pairs of points from 2 input matrixes, using the HEOM metric, that way categorical attributes are
        allow in the dataset.
        --------
        Parameters: 
        X (numpy.array): An (N*M) numpy matrix containing the first set of points, where N is the number of points and M is the number of attributes per point.
        X2 (numpy.array): An (N*M) numpy matrix containing the second set of points, where N is the number of points and M is the number of attributes per point.
        --------
        Returns:
        dist_matrix (numpy.array): A (M*M) matrix containing the distance between all pairs of points in X
        '''
        meta= self.meta
        
        dist_matrix=np.zeros((len(X2),len(X)))
        

        #calculate the ranges of all attributes
        range_max=np.max(X,axis=1)
        range_min=np.min(X,axis=1)

        for i in range(len(X2)): 
            for j in range(len(X)):
                #for attribute
                dist = 0
                for k in range(len(X2[0])):
                    #missing value
                    if(X2[i][k] == None or X[j][k]==None):
                        dist+=1
                    #numerical
                    if(meta[k]==0):
                        
                        if(range_max[k]-range_min[k]==0):
                            dist+=(abs(X2[i][k]-X[j][k]))**2
                        else:
                            dist+=(abs(X2[i][k]-X[j][k])/(range_max[k]-range_min[k]))**2
                            
                    #categorical
                    if(meta[k]==1):
                        if(X2[i][k]!=X[j][k]):
                            dist+=1
                dist_matrix[i][j]=np.sqrt(dist)

        #print(dist_matrix)
        return dist_matrix

    




   
    def __calculate_distance_matrix(self,X,distance_func="HEOM"):
        '''
        Is called by the __init__ method.
        Function used to select which distance metric will be used to calculate the distance between a matrix of points.
        Only the HEOM metric is implemented for now, however if more metrics are added this function can easily be changed to
        incomporate the new metrics.
        --------
        Parameters:
        X (numpy.array): An (N*M) numpy matrix containing the points, where N is the number of points and M is the number of attributes per point.
        distance_func (string): The distance function to be used, only available option right now is "HEOM"
        --------
        Returns:
        dist_matrix (numpy.array): A (M*M) matrix containing the distance between all pairs of points in X
        --------
        '''
        if(distance_func=="HEOM"):
            distance_matrix,unnorm_distance_matrix=self.__distance_HEOM(X)
        elif(distance_func=="default"):
            distance_matrix,unnorm_distance_matrix=self.__distance_HEOM(X)
        
        #add other distance functions

        return distance_matrix,unnorm_distance_matrix

    



    def __get_class_inxs(self):
        '''
        Called by the __init__ method.
        Calculates what are the indexes in X and y for each of the classes.
        --------
        Returns:
        class_inds (array): An array of arrays where each inner array contains the indexes for one class. The number of inner arrays is equal to the
        total number of unique classes in X.
        --------
        '''
        class_inds = []
        for cls in self.classes:
            cls_ind=np.where(self.y==cls)[0]
            class_inds.append(cls_ind)
        return class_inds


    def __knn(self,inx,line,k,clear_diag=True):
        '''
        Called by all the complexity metrics that need to calculate the nearest neighbours of a sample.
        Calculates the class labels of the k nearest neighbours of a sample x. If clear_diag is True, it is assumed that point in
        position "inx" is the sample x, which will have distance 0 to itself and so the distance is changed to infinite to avoid
        one of the nearest neighbours being the point itself.
        --------
        Parameters:
        inx (int): The index of the point in the array line. Not relevant if "line" does not contain the point itself.
        line (array): An array of distances from "x" to all the points. Usually taken from the dist_matrix attribute of the class.
        k (int): the number of neighbours to consider.
        clear_diag (bool): True if the distance in position "inx" of "line" is the point itself.
        --------
        Returns:
        count: An (n*1) array, where n is the number of unique class labels, where in each position is the value of how many of the k
        nearest neighbours belong to that class. For example if k=5 and there are 3 classes a possible configuration could be [2,3,0] meaning that 2
        of the neighbours are from the 1st class, 3 are from the second class, and there are none from the 3rd class.
        '''
        count = np.zeros(len(self.classes))
        if(clear_diag):
            line[inx]=math.inf
        for i in range(k):
            index=np.where(line == min(line))[0][0]
            line[index]=math.inf
            #if(self.y[index]!=self.y[inx]):
            cls_inx = np.where( self.classes == self.y[index])[0][0]  
            count[cls_inx]+=1

        return count
    

    def __knn_dists(self,inx,line,k,clear_diag=True):
        dists = []
        if(clear_diag):
            line[inx]=math.inf
        for i in range(k):
            index=np.where(line == min(line))[0][0]
            if(self.y[index]==self.y[inx]):
                dists.append(line[index])
         
            line[index]=math.inf
            #if(self.y[index]!=self.y[inx]):
        
        return dists


    def __hypersphere(self,inx,sigma,distance_matrix=[],y=[]):
        '''
        Called by the C1 and MRCA complexity measures.
        Draws an hypersphere of radius "sigma" and center on a sample x and calculates the number of samples that 
        are from the same class as a sample "x" as well as the number of sample that are not from the same class.
        The sample will count itself as a instance of the same class inside the hypersphere.
        If no distance matrix or class label arrays are passed the function assumes uses the attributes of the complexity
        class calcultated in the __init__ method.
        --------
        Parameters:
        inx (int): Index of the sample x in the array X
        sigma (float): Size of the radius of the hypersphere
        distance_matrix (numpy.array): An (N*N) matrix, where N is the number of points in the dataset, contains the pair wise
        distances between all points
        y (numpy.array): An (N*1) array containing all the class labels from the dataset
        --------
        Returns:
        n_minus (int): The number of instances that have a different class label than sample x
        n_plus (int): The number of instances that have the same class label than sample x
        --------
        '''
     
        if(len(distance_matrix)==0):
          
            distance_matrix=self.dist_matrix
          
        if(len(y)==0):
            y=self.y
        

       
        #an array containing the distance to all points
        line = distance_matrix[inx]
        
        n_minus = 0
        n_plus = 0
        for i in range(len(line)):
            #if the sample is inside de hypersphere
           
            #print(line)
            if(line[i]<=sigma):
                #if the sample is from the same class as "x"
                if(y[i]==y[inx]):
                    n_plus+=1
                else:
                    n_minus+=1
        return [n_minus,n_plus]

   
    def __hypersphere_sim(self,inx,sigma):
        '''
        Called by C2 function.
        Similiar to the hypersphere function, however instead of calculating the number of samples that are from the same class and the number
        of samples that isn't, it calculates the sum of the distance from sample "x" to all samples of with the same class label and the sum of distance 
        to all samples with a different class label.
        --------
        Parameters: 
        inx (int): The index of the sample "x" in the array X
        sigma (float): The radius of the hypersphere
        --------
        Returns:
        n_minus: The sum of the distances from "x" to all the samples inside the hypersphere with a different class label
        n_plus: The sum of the distances from "x" to all the samples inside the hypersphere with the same class label
        --------
        '''
        line = self.dist_matrix[inx]
        n_minus = 0
        n_plus = 0
        for i in range(len(line)):
            #if the sample is inside the hypersphere
            if(line[i]<=sigma):
                #if the sample is from the same class as "x"
                if(self.y[i]==self.y[inx]):
                    n_plus+=line[i]
                else:
                    n_minus+=line[i]
        return [n_minus,n_plus]


    
    def R_value(self,k=5,theta=2):
        '''
        Calculate the Augmented R value complexity measure defined in [1].

        --------
        Parameters:
        k (int): Number of neighbours, for the knn function
        tetha (int): threshold of neighbours that can have a different class label
        --------
        Returns:
        r_values (array): An array with the R values of the pair wise combinations of all classes (One VS One)
        --------
        References:

        [1] Borsos Z, Lemnaru C, Potolea R (2018) Dealing with overlap and imbalance:
        a new metric and approach. Pattern Analysis and Applications 21(2):381-395
        --------
        '''
        r_matrix=np.zeros((len(self.classes),len(self.classes)))

        for i in range(len(self.dist_matrix)):
            line = self.dist_matrix[i]
            #caclutate the k nearest neighbours of the instance
            count=self.__knn(i,copy.copy(line),k)
            cls_inx = np.where( self.classes == self.y[i])[0][0]
            for j in range(len(self.classes)):
                #check if the threshold of neighbours is crossed
                if(theta<count[j]):
                   
                    r_matrix[cls_inx,j]+=1

        #print(r_matrix)
        
        for i in range(len(r_matrix)):
            for j in range(len(r_matrix[0])):
                r_matrix[i,j]=r_matrix[i,j]/self.class_count[i]


        #print(r_matrix)

        r_values = []
        for i in range(len(r_matrix)):
            for j in range(i+1,len(r_matrix)):
                imbalanced_ratio = 0
                maj_r = 0
                min_r = 0
                if(self.class_count[i]>self.class_count[j]):
                    imbalanced_ratio = self.class_count[j]/self.class_count[i]
                    maj_r = r_matrix[i,j]
                    min_r = r_matrix[j,i]
                else:
                    imbalanced_ratio = self.class_count[i]/self.class_count[j]
                    maj_r = r_matrix[j,i]
                    min_r = r_matrix[i,j]

                r=(1/(imbalanced_ratio+1))*(maj_r+imbalanced_ratio*min_r)
                r_values.append(r)
        
        
        return r_values
       
    def D3_value(self,k=5):
        '''
        Calculate the D3 value complexity measure defined in [1].

        --------
        Parameters:
        k (int): Number of neighbours, for the knn function
        --------
        Returns:
        d3_matirx (np.array): An array with the d3 values for all classes
        --------
        References:

        [1] Sotoca JM, Mollineda RA, Sanchez JS (2006) A meta-learning framework
        for pattern classication by means of data complexity measures. Inteligencia
        Articial Revista Iberoamericana de Inteligencia Artificial 10(29):31-38
        --------
        '''
        d3_matrix=np.zeros(len(self.classes))
    
        for i in range(len(self.dist_matrix)):
            line = self.dist_matrix[i]
            count=self.__knn(i,copy.copy(line),k)
            cls_inx = np.where( self.classes == self.y[i])[0][0]
            if(0.5>(count[cls_inx]/k)): 
                d3_matrix[cls_inx]+=1 
        return d3_matrix


    
    def kDN(self,k=5):
        '''
        Calculate the kDN value complexity measure defined in [1]

        --------
        Parameters:
        k (int): Number of neighbours, for the knn function
        --------
        Returns:
        kDN (float): The value of the complexity measure
        --------
        References:

        [1] Smith MR, Martinez T, Giraud-Carrier C (2014) An instance level analysis
        of data complexity. Machine learning 95(2):225-256
        --------
        '''


        kDN_value = 0
        for i in range(len(self.dist_matrix)):
            line = self.dist_matrix[i]
            count=self.__knn(i,copy.copy(line),k)
            cls_inx = np.where( self.classes == self.y[i])[0][0]
            kDN_value+= (k-count[cls_inx])/k
        kDN_value/=len(self.X)       
        return kDN_value

    def CM(self,k=5):
        '''
        Calculate the CM value complexity measure defined in [1].

        -------
        Parameters:
        k (int): Number of neighbours, for the knn function
        -------
        Returns:
        CM_value (float): The value of the complexity measure
        -------
        References:

        [1] Anwar N, Jones G, Ganesh S (2014) Measurement of data complexity
        for classification problems with unbalanced data. Statistical Analysis and
        Data Mining: The ASA Data Science Journal 7(3):194-211
        -------
        '''
        CM_value = 0
        for i in range(len(self.dist_matrix)):
            line = self.dist_matrix[i]
            count=self.__knn(i,copy.copy(line),k)
            cls_inx = np.where( self.classes == self.y[i])[0]
            kDN_value = (k-count[cls_inx])/k
            if(kDN_value>0.5):
                CM_value+=1
        CM_value/=len(self.X)
        return CM_value   

    

    def __MRI_p(self,profile):
        '''
        Calculate the MRI value of a pattern.

        ------
        Parameters:
        profile (array): An array containing the features of the pattern.
        ------
        Returns:
        mri_val (float): The MRI value of the pattern.
        
        '''
        sum_val = 0
        m = len(profile)
        for j in range(m):
            w = (1-(j/m))
            sum_val += w * (1-profile[j])
        mri_val = sum_val*(1/(2*m))
        #print(profile)
        #print(mri_val)
        return mri_val

    def __MRI_k(self,cluster):
        '''
        Called by the MRCA function. Calculates the Multiresolution Index (MRI) for a cluster by
        averaging the MRI value of each pattern in the cluster.
        --------
        Parameters:
        cluster (array): An array of arrays (N*M) containg all the points in a cluster, where N is the number of points and M is the 
        dimesion of each point.
        ---------
        Returns:
        mri_val (float): The multiresolution index of the cluster
        ---------
        '''
        sum_val = 0
        for profile in cluster:
            sum_val+= self.__MRI_p(profile)
        mri_val=sum_val/len(cluster)
        return mri_val

    def MRCA(self,sigmas=[0.25,0.5,0.75],n_clusters=3,distance_func="default"):
        '''
        Calculates the MRCA value complexity measure defined in [1].

        -------
        Parameters:
        sigmas (float): An array with the multiple hypersphere radius
        n_clusters (int): the number of clusters to group in the kMeans algorithm
        -------
        Returns:
        mrca (array): An array (n_clusters*1), with the mrca value of each cluster
        -------
        References:

        [1] Armano G, Tamponi E (2016) Experimenting multiresolution analysis for
        identifying regions of different classification complexity. Pattern Analysis
        and Applications 19(1):129-137
        '''

        #one vs one approach
        for i2 in range(len(self.class_inxs)):
            for j2 in range(i2+1,len(self.class_inxs)):

                #create new arrays with just the 2 classes being considired for this iteration
                c1 = self.classes[i2]
                c2 = self.classes[j2]
                sample_c1 = self.X[self.class_inxs[i2]]
                sample_c2 = self.X[self.class_inxs[j2]]
                sample_c1_y = self.y[self.class_inxs[i2]]
                sample_c2_y = self.y[self.class_inxs[j2]]
                new_X = np.concatenate([sample_c1,sample_c2],axis=0)
                new_y = np.concatenate([sample_c1_y,sample_c2_y],axis=0)
                new_dist_matrix,_ = self.__calculate_distance_matrix(new_X,distance_func=distance_func)
                
                #print(new_dist_matrix)
                
                mrca=np.zeros(n_clusters)
                profiles = np.zeros((len(new_X),len(sigmas)))
                

                #for each point
                for i in range(len(new_X)):
                    #for each radius
                    for j in range(len(sigmas)):
                        sigma = sigmas[j]
                        #change this: if there is more than 2 classes the hypershpere n_minus of all other classes
                        
                        
                        n = self.__hypersphere(i,sigma,distance_matrix=new_dist_matrix,y=new_y)
                        

                       
                        #calculate the psi values for each profile in accordance to [1]
                        if(new_y[i]==c1):
                            alt_y = 1
                            psi = alt_y * ((n[1]-n[0])/(n[1]+n[0]))      
                        else:
                            alt_y = -1
                            psi = alt_y * ((n[0]-n[1])/(n[0]+n[1]))      
                        profiles[i,j]=psi

                #cluster the the profiles
                print(profiles)
                kmeans = KMeans(n_clusters=n_clusters).fit(profiles)
                

                #for each cluster calculate the MRI value
                for i in range(n_clusters):
                    inx = np.where( kmeans.labels_ == i)[0]
                    cluster = profiles[inx]
                    
                    mrca[i]=self.__MRI_k(cluster)

                return mrca
                
    def C1(self,max_k=5):
        '''
        Calculate the C1 value complexity measure defined in [1].

        -------
        Parameters:
        sigmas (float): An array with the multiple hypersphere radius
        -------
        Returns:
        c1_val (float): The value of the complexity measure 
        -------
        References:

        [1] Massie S, Craw S, Wiratunga N (2005) Complexity-guided case discovery
        for case based reasoning. In: AAAI, vol 5, pp 216-221
        '''
        c1_sum = 0
        for i in range(len(self.X)):
            c1_instance_sum = 0
            cls_inx = np.where( self.classes == self.y[i])[0][0]
            #for each radius sigma
            for k in range(1,max_k+1):
                #draw a hypersphere with radius sigma around the point and check which points inside are from the same class and
                #which are not.
                #n = self.__hypersphere(i,sigma)
                
                count=self.__knn(i,copy.copy(self.dist_matrix[i]),k)
               
                pkj=count[cls_inx]/k
                c1_instance_sum += pkj

            c1_instance_val = 1-(c1_instance_sum/max_k)
            c1_sum += c1_instance_val
        c1_val = c1_sum/len(self.X)

        return c1_val

    def C1_imb(self,max_k=5):
        '''
        Calculate the C1 value complexity measure defined in [1].

        -------
        Parameters:
        sigmas (float): An array with the multiple hypersphere radius
        -------
        Returns:
        c1_val (float): The value of the complexity measure 
        -------
        References:

        [1] Massie S, Craw S, Wiratunga N (2005) Complexity-guided case discovery
        for case based reasoning. In: AAAI, vol 5, pp 216-221
        '''

        c1_sum = np.zeros(len(self.classes))
        for i in range(len(self.X)):
            c1_instance_sum = 0
            cls_inx = np.where( self.classes == self.y[i])[0][0]
            #for each radius sigma
            for k in range(1,max_k+1):
                #draw a hypersphere with radius sigma around the point and check which points inside are from the same class and
                #which are not.
                #n = self.__hypersphere(i,sigma)
                
                count=self.__knn(i,copy.copy(self.dist_matrix[i]),k)
               
                pkj=count[cls_inx]/k
                c1_instance_sum += pkj

            c1_instance_val = 1-(c1_instance_sum/max_k)
            c1_sum[cls_inx] += c1_instance_val
        c1_val = np.divide(c1_sum,self.class_count)

        return c1_val
    
    


    def C2(self,max_k=5):
        c2_sum = 0
        for i in range(len(self.X)):
            c2_instance_sum = 0

            #for each radius sigma
            for k in range(1,max_k+1):
                #draw a hypersphere with radius sigma around the point and check which points inside are from the same class and
                #which are not.
                #n = self.__hypersphere(i,sigma)
                
                dists=self.__knn_dists(i,copy.copy(self.dist_matrix[i]),k)
            
                pkj = 0
                for d in dists:
                    pkj+=1-d
                pkj/= k

                #cls_inx = np.where( self.classes == self.y[i])[0][0]
                #pkj=count[cls_inx]/k
                c2_instance_sum += pkj

            c2_instance_val = 1-(c2_instance_sum/max_k)
            c2_sum += c2_instance_val
        c2_val = c2_sum/len(self.X)

        return c2_val


    def C2_imb(self,max_k=5):
        c2_sum = np.zeros(len(self.classes))
        for i in range(len(self.X)):
            c2_instance_sum = 0
            cls_inx = np.where( self.classes == self.y[i])[0][0]
            #for each radius sigma
            for k in range(1,max_k+1):
                #draw a hypersphere with radius sigma around the point and check which points inside are from the same class and
                #which are not.
                #n = self.__hypersphere(i,sigma)
                
                dists=self.__knn_dists(i,copy.copy(self.dist_matrix[i]),k)
            
                pkj = 0
                for d in dists:
                    pkj+=1-d
                pkj/= k

                #cls_inx = np.where( self.classes == self.y[i])[0][0]
                #pkj=count[cls_inx]/k
                c2_instance_sum += pkj

            c2_instance_val = 1-(c2_instance_sum/max_k)
            c2_sum[cls_inx] += c2_instance_val
        c2_val = np.divide(c2_sum,self.class_count)

        return c2_val



    def __calculate_n_inter(self,dist_matrix=[],y=[],imb=False):
        '''
        Called by the DBC and N1 complexity measure functions.
        Calculates the miminimum spanning tree using a distance matrix, dist_matrix. Afterwards it counts the amount
        of vertixes in the MST that have an edge connecting them are from 2 distinct classes.
        If no distance matrix or class label array y is passed it assumes the attributes calculated in the
        __init__ method.
        -------
        Parameters:
        dist_matrix (numpy.array): A distance matrix between all points, used to calculate the MST
        y (numpy.array): an array with the class labels
        -------
        Returns:
        count (int): the number of vertixes connected by an edge that have different class labels
        '''


        #If no parameters are passed it uses the distance matrix and class labels calcultated in the __init__ method
        #This is necessary because the distance matrix and class labels are different depending on the complexity measure
        #that calls this function.
        if len(dist_matrix)==0:
            dist_matrix = self.dist_matrix
        if len(y) == 0:
            y = self.y
       
        #calculate the MST using the distance matrix
        minimum_spanning_tree = scipy.sparse.csgraph.minimum_spanning_tree(csgraph=np.triu(dist_matrix, k=1), overwrite=True)
        

        #convert the mst to an array
        mst_array = minimum_spanning_tree.toarray().astype(float)
        
      
        #iterate over the MST to determine which vertixes that are connected are from different classes.
        vertix = []
        aux_count = 0
        for i in range(len(mst_array)):
            for j in range(len(mst_array[0])):
                if(mst_array[i][j]!=0):
                    
                    
                    if (y[i]!=y[j]):
                        
                        vertix.append(i)
                        vertix.append(j)

        unique_vertix = np.unique(vertix)

        #---------------------------

        
        #----------------------------
        if(imb==False):
            count=len(unique_vertix)

            
        else:
            count = np.zeros(len(self.classes))
            for inx in unique_vertix:
                cls_inx = np.where( self.classes == self.y[inx])[0][0]
                count[cls_inx]+=1


        return count

    def N1_imb(self):
        count = self.__calculate_n_inter(imb=True)
        #print(count)
        n1 = np.divide(count,self.class_count)
        return n1

    def N1(self):
        '''
        Calculate the N1 value complexity measure defined in [1].

        -------
        Returns:
        n1 (float): The value of the complexity measure 
        -------
        References:

        Ho T, Basu M (2002) Complexity measures of supervised classification
        problems. IEEE transactions on pattern analysis and machine intelligence 24(3):289-300
        '''
        count = self.__calculate_n_inter()
        #print(count)
        n1 = count/len(self.y)
        return n1
    
    def N2_imb(self):
        count_inter = np.zeros(len(self.classes))
        count_intra = np.zeros(len(self.classes))


        #for each sample
        for i in range(len(self.dist_matrix)):
            min_inter = np.inf
            min_intra = np.inf

            #iterate over every sample and check which is the nearest neighbour from the same class and
            #which is the nearest neighbour from the opposite class.
            for j in range(len(self.dist_matrix[0])):
                if(self.y[i]==self.y[j] and i!=j and self.dist_matrix[i][j]<min_intra):
                    min_intra=self.dist_matrix[i][j]
                if(self.y[i]!=self.y[j] and self.dist_matrix[i][j]<min_inter):
                    min_inter=self.dist_matrix[i][j]

            cls_inx = np.where( self.classes == self.y[i])[0][0]
            count_inter[cls_inx]+=min_inter
            count_intra[cls_inx]+=min_intra
        

        
        r = np.array([])
        for i in range(len(count_inter)):
            if(count_inter[i]==0):
                r=np.append(r,0)
            else:
                r=np.append(r,(count_intra[i]/count_inter[i]))
        
        print(r)
        N2 = np.divide(r,(1+r))
        return N2

    def N2(self):
        '''
        Calculate the N2 value complexity measure defined in [1].

        -------
        Returns:
        n2 (float): The value of the complexity measure 
        -------
        References:

        Ho T, Basu M (2002) Complexity measures of supervised classification
        problems. IEEE transactions on pattern analysis and machine intelligence 24(3):289-300
        '''
        count_inter = 0
        count_intra = 0


        #for each sample
        for i in range(len(self.dist_matrix)):
            min_inter = np.inf
            min_intra = np.inf

            #iterate over every sample and check which is the nearest neighbour from the same class and
            #which is the nearest neighbour from the opposite class.
            for j in range(len(self.dist_matrix[0])):
                if(self.y[i]==self.y[j] and i!=j and self.dist_matrix[i][j]<min_intra):
                    min_intra=self.dist_matrix[i][j]
                if(self.y[i]!=self.y[j] and self.dist_matrix[i][j]<min_inter):
                    min_inter=self.dist_matrix[i][j]
            count_inter+=min_inter
            count_intra+=min_intra


        if(count_inter==0):
            r = 0
        else:
            r = count_intra/count_inter

        
        

        N2 = r/(1+r)
        return N2

   
    def N3(self,k=1):
        '''
        Calculate the N3 value complexity measure in [1].
        By default k=1 in accordance to the definition in the paper. However it is possible to select a different value for k.
        -------
        Parameters:
        k (int): number of nearest neighbours to consider
        -------
        Returns:
        n3 (float): the value of the complexity measure 
        -------
        References: 

        Ho T, Basu M (2002) Complexity measures of supervised classification
        problems. IEEE transactions on pattern analysis and machine intelligence 24(3):289-300
        ''' 
       
        sample_count=0
        #for each sample
        for sample in range(len(self.X)):
            #calculate the nearest neighbour
            count=self.__knn(sample,copy.copy(self.dist_matrix[sample]),k)
            cls_inx = np.where( self.classes == self.y[sample])[0][0]
            
            
            class_count=count[cls_inx]
            max_count=np.max(count)
            #are there more instances with the same class label or with a different class label. 
            if(class_count<max_count):
                sample_count+=1


        n3 = sample_count/len(self.y)
        return n3
    
    #todo: change comments
    def N3_imb(self,k=1):
        '''
        Calculate the N3 value complexity measure in [1].
        By default k=1 in accordance to the definition in the paper. However it is possible to select a different value for k.
        -------
        Parameters:
        k (int): number of nearest neighbours to consider
        -------
        Returns:
        n3 (float): the value of the complexity measure 
        -------
        References: 

        Ho T, Basu M (2002) Complexity measures of supervised classification
        problems. IEEE transactions on pattern analysis and machine intelligence 24(3):289-300
        ''' 
        n3_counts=np.zeros(len(self.classes))
        #sample_count=0
        #for each sample
        for sample in range(len(self.X)):
            #calculate the nearest neighbour
            count=self.__knn(sample,copy.copy(self.dist_matrix[sample]),k)
            cls_inx = np.where( self.classes == self.y[sample])[0][0]
            
            
            class_count=count[cls_inx]
            max_count=np.max(count)
            #are there more instances with the same class label or with a different class label. 
            if(class_count<max_count):
                #sample_count+=1
                n3_counts[cls_inx]+=1
        n3 = np.divide(n3_counts,self.class_count)
        #n3 = sample_count/len(self.y)
        return n3
    
    def __interpolate_samples(self,class_inxs=[]):
        '''
        Create new interpolated samples from the sample array X.
        To achieve this 2 samples in X from the same class are selected and a new sample is created by interpolating these 2.
        This sample will have the same class label of the 2 used to create it.
        This process is repeated N times, where N is the size of X. 
        ------
        Returns:
        X_interp (array): The new array with the interpolated samples
        y_inter (array): An array with the labels of the new class samples.
        '''

        if(len(class_inxs)==0):
            class_inxs = self.class_inxs
        
        X_interp = []
        y_interp = []
        for cls_inx in class_inxs:
            new_X = self.X[cls_inx,:]
            new_y = self.y[cls_inx]
            sample1_inxs = np.random.choice( len(new_X),  len(new_X))
            sample2_inxs = np.random.choice( len(new_X),  len(new_X))
            sample1 = new_X[sample1_inxs, :]
            sample2 = new_X[sample2_inxs, :]

            alpha = np.random.ranf(new_X.shape)

            X_interp_cls = sample1 + (sample2 - sample1)*alpha

            
            y_interp=np.append(y_interp,new_y)
            if(len(X_interp)==0):
                X_interp=X_interp_cls
            else:
                X_interp=np.concatenate((X_interp,X_interp_cls),axis=0)
        
       
        return X_interp, y_interp

    
    
    
    
    
    
    
   
    def N4(self,k=1):
        '''
        Calculate the N4 value complexity measure defined in [1].
        -------
        Parameters:
        k (int): Number of nearest neighbours to consider
        -------
        Returns:
        n4 (float): The value of the complexity measure 
        -------
        References:

        [1] Lorena AC, Garcia LP, Lehmann J, Souto MC, Ho TK (2019) How complex
        is your classification problem? a survey on measuring classification
        complexity. ACM Computing Surveys (CSUR) 52(5):1-34
        '''


        #create new interpolated samples
        X_interp, y_interp=self.__interpolate_samples()
        
       
       

        new_dist = self.__distance_HEOM_different_arrays(self.X,X_interp)

        sample_count=0

        #for each sample in X_interpol check calculate their k nearest neighbours in X and determine if there
        #are more neighbours from the same class or more neighbours from the opposite classes.
        for sample in range(len(X_interp)):
            
            
            count=self.__knn(sample,copy.copy(new_dist[sample]),k,clear_diag=False)
            cls_inx = np.where( self.classes == y_interp[sample])[0][0]
            
            #number of neighbours with the same class label
            class_count=count[cls_inx]

            
            max_count=np.max(count)
            if(class_count<max_count):
                sample_count+=1
                

        n4 = sample_count/len(y_interp)
        return n4
    

    #todo change this
    def N4_imb(self,k=1):
        '''
        Calculate the N4 value complexity measure defined in [1].
        -------
        Parameters:
        k (int): Number of nearest neighbours to consider
        -------
        Returns:
        n4 (float): The value of the complexity measure 
        -------
        References:

        [1] Lorena AC, Garcia LP, Lehmann J, Souto MC, Ho TK (2019) How complex
        is your classification problem? a survey on measuring classification
        complexity. ACM Computing Surveys (CSUR) 52(5):1-34
        '''


        #create new interpolated samples
        X_interp, y_interp=self.__interpolate_samples()
        
       
       

        new_dist = self.__distance_HEOM_different_arrays(self.X,X_interp)

        #sample_count=0
        n4_counts=np.zeros(len(self.classes))
        #for each sample in X_interpol check calculate their k nearest neighbours in X and determine if there
        #are more neighbours from the same class or more neighbours from the opposite classes.
        for sample in range(len(X_interp)):
            
            
            count=self.__knn(sample,copy.copy(new_dist[sample]),k,clear_diag=False)
            cls_inx = np.where( self.classes == y_interp[sample])[0][0]
            
            #number of neighbours with the same class label
            class_count=count[cls_inx]

            
            max_count=np.max(count)
            if(class_count<max_count):
                #sample_count+=1
                n4_counts[cls_inx]+=1
        n4 = np.divide(n4_counts,self.class_count)

        #n4 = sample_count/len(y_interp)
        return n4
        
       
    
    def SI(self,k=1):
        '''
        Calculate the Separability index (SI) complexity measure defined in [1].

        --------
        Returns:
        si_measure (float): The SI complexity measure value calculated.
        --------
        References:

        Thornton C (1998) Separability is a learner's best friend. In: 4th Neural
        Computation and Psychology Workshop, London, 9-11 April 1997,
        Springer, pp 40-46


        
        '''
        

        sample_count=0
        #for each sample
        for sample in range(len(self.X)):
            #calculate the nearest neighbour
            count=self.__knn(sample,copy.copy(self.dist_matrix[sample]),k)
            cls_inx = np.where( self.classes == self.y[sample])[0][0]
            
            
            class_count=count[cls_inx]
            max_count=np.max(count)
            #are there more instances with the same class label or with a different class label. 
            if(class_count==max_count):
                sample_count+=1


        
        si_measure = sample_count/len(self.y)
        return si_measure
        



    def __find_nearest_oposite_class(self,x_inx,x_dist):
        '''
        Function called by __find_nearest_oposite_class_all.
        Finds the nearest sample of the opposite class label for a sample
        ------
        Parameters:
        x_inx (int): The index of the sample in X
        x_dist (array): An array with the distances from the sample to every other point
        ------
        Returns:
        nearest_oposite_class_inx (int): The index of the nearest sample of the opposite class
        nearest_oposite_class_dist (float): The distance to the nearest sample of the opposite class
        '''
        nearest_oposite_class_dist = np.inf
        nearest_oposite_class_inx = None
        for i in range(len(x_dist)):
            if (x_dist[i]<nearest_oposite_class_dist and self.y[x_inx]!=self.y[i]):
                nearest_oposite_class_dist = x_dist[i]
                nearest_oposite_class_inx = i
        return nearest_oposite_class_inx,nearest_oposite_class_dist
    
    def __find_nearest_oposite_class_all(self,dist_matrix=[]):
        '''
        Function called by  __get_sphere_count, LSC and Clust.
        Find the nearest sample of an opposite class label for every sample in X.
        -------
        Parameters:
        dist_matrix (numpy.array): An (N*N) matrix with the pair wise distances between all samples
        -------
        Returns:
        nearest_oposite_class_array (numpy.array): an array with the indexes of the nearest sample of the opposite class for every sample in X
        nearest_oposite_class_dist_array (numpy.array): an array with the distances of the nearest sample of the opposite class for every sample in X
        '''
        if(len(dist_matrix)==0):
            dist_matrix=self.dist_matrix

        nearest_oposite_class_array=[]
        nearest_oposite_class_dist_array=[]
        for i in range(len(dist_matrix)):
            nearest_oposite_class_inx,nearest_oposite_class_dist=self.__find_nearest_oposite_class(i,dist_matrix[i])
            nearest_oposite_class_array.append(nearest_oposite_class_inx)
            nearest_oposite_class_dist_array.append(nearest_oposite_class_dist)
        return np.array(nearest_oposite_class_array),np.array(nearest_oposite_class_dist_array)
    
    
    
    def __find_spheres(self,ind,e_ind,e_dist,radius):
        '''
        Called by __get_sphere_count. 
        Calculates the radius of every hypersphere as defined by the T1 metric.
        -----
        Parameters:
        ind (int): The index of the sample to calculate the radius for.
        e_ind (int): The index of the nearest sample of an opposite class.
        e_dist (float): The distance to the nearest sample of an opposite class.
        radius (numpy.array): An array with the radius of every sample. This array starts with every position as -1
        and will be filled with the true radius with every recursive call.
        -----
        Returns:
        radius[ind] (float): the radius of the sample in index "ind".
        '''


        if radius[ind] >= 0.0:
            return radius[ind]

        ind_enemy = e_ind[ind]


        #stop condition, the both samples are each other's nearest neighbour
        if(ind == e_ind[ind_enemy]):
            
            radius[ind_enemy] = 0.5 * e_dist[ind]
            radius[ind]  = 0.5 * e_dist[ind]
            
            return radius[ind]
        #give a temporary value
        radius[ind] = 0.0
        

        #find the radius for the nearest of the opposite class 
        radius_enemy = self.__find_spheres(ind_enemy,e_ind,e_dist,radius)

        #knowing the radius of the nearest of the opposite class, calculate the radius of this sample
        radius[ind] = abs(e_dist[ind] - radius_enemy)

        return radius[ind]


    
    def __is_inside(self,center_a,center_b,radius_a,radius_b):
        '''
        Check if a hypersphere a is inside an hypersphere b.

        -----
        Parameters:
        center_a (array): An array containing the center of hypersphere a
        center_b (array): An array containing the center of hypersphere b
        radius_a (float): the radius of hypersphere a
        radius_b (float): the radius of hypersphere b
        ------
        Returns:
        var (bool): True if hypersphere a is inside hypersphere b, false if not.
        '''
        var = False
        '''
        print("--------------")
        print(center_a)
        print(radius_a)
        print(center_b)
        print(radius_b)
        print(np.sqrt(sum(np.square(center_a-center_b))) - (radius_b-radius_a))
        print("------------")
        '''
       

       
        if(abs(np.sqrt(sum(np.square(center_a-center_b))) - (radius_b-radius_a))<0.01):
            
            var=True
        return var

    def _scale_N(self,N: np.ndarray) -> np.ndarray:
        """Scale all features of N to [0, 1] range."""
        N_scaled = N

        if not np.allclose(1.0, np.max(N, axis=0)) or not np.allclose(
            0.0, np.min(N, axis=0)
        ):
            N_scaled = sklearn.preprocessing.MinMaxScaler(
                feature_range=(0, 1)
            ).fit_transform(N)

        return N_scaled  
    
    def __remove_overlapped_spheres(self,radius):
        '''
        Remove all the hyperspheres that are completely contained inside another.
        ------
        Parameters: 
        radius (numpy.array): An array containing the radius of all the hyperspheres.
        ------
        Returns:
        sphere_inst_num (numpy.array): An array containing the number of hyperspheres completely inside each hypersphere, if during the execution
        of this function an hypersphere was found to be completly inside another then its count will be 0.
        '''
        
        X = self._scale_N(self.X)
        #X = self.X
        inx_sorted = np.argsort(radius)
        inst_per_sphere = np.ones(len(self.X), dtype=int)

        for inx1, inx1_sphere in enumerate(inx_sorted[:-1]):
            
           
            for inx2_sphere in inx_sorted[:inx1:-1]:
                
                if (self.__is_inside(X[inx1_sphere],X[inx2_sphere],radius[inx1_sphere],radius[inx2_sphere])):
                    #print("FOUND")
                    #print(X[inx1_sphere])
                    #print(X[inx2_sphere])
                    inst_per_sphere[inx2_sphere] += inst_per_sphere[inx1_sphere]
                    inst_per_sphere[inx2_sphere] = 0
                    break

        return inst_per_sphere    

    


    def __get_sphere_count(self):
        '''
        Called by the T1, NSG and ICSV function.
        Calculates the number of samples inside each hypersphere as well as each hypersphere radius, in accordance to
        the T1 measure.
        ------
        Returns:
        sphere_inst_count (numpy.array): An (1*N) with the number of samples inside each hypersphere, where N is the number of samples
        radius (numpy.array): An (1*N) with the radius of each hypersphere, where N is the number of samples.
        ------
        '''

        #find the nearest sample of the opposite class for every sample in X.
        e_ind,e_dist = self.__find_nearest_oposite_class_all(dist_matrix=self.dist_matrix)
        #print(self.X)
        #print(self.unnorm_dist_matrix)
        #print(e_dist)
        #print(nearest_enemy_dist)
        
        
        radius = np.array([-1.0]*len(e_ind))
        #print(nearest_enemy_ind)


        #calculates the radius of each hypersphere
        for ind in range(len(radius)):
            if radius[ind] < 0.0:
                self.__find_spheres(ind,e_ind,e_dist,radius)
        
        
        sphere_inst_count = np.array([])
        ordered_radius = np.array([])

        #remove the hyperspheres that are complety inside another hypersphere.
        sphere_inst_count=self.__remove_overlapped_spheres(radius)
        #print(scaled_X)
        #print(radius)
        '''
        for i in range(len(self.class_inxs)):
            sphere_inst_count = np.concatenate((sphere_inst_count,self.__remove_overlapped_spheres(scaled_X[self.class_inxs[i],:],radius[self.class_inxs[i]])))   
            ordered_radius = np.concatenate((ordered_radius,radius[self.class_inxs[i]]))
        '''
        #print(sphere_inst_count)
        
        #print(radius)
        return sphere_inst_count,radius
    

    #only for datasets with no categorical features
    def T1(self):
        '''
        Calculate the T1 value complexity measure defined in [1].

        -------
        Returns: 
        t1 (float): The t1 complexity measure
        -------
        References:

        [1] Lorena AC, Garcia LP, Lehmann J, Souto MC, Ho TK (2019) How complex
        is your classification problem? a survey on measuring classification
        complexity. ACM Computing Surveys (CSUR) 52(5):1-34
        '''
        sphere_inst_count,radius=self.__get_sphere_count()
        print(radius)
        t1 = len(sphere_inst_count[sphere_inst_count > 0])/ len(self.y)
        
        #t1 = sum(sphere_inst_count[sphere_inst_count > 0]) / len(self.y)))/len(sphere_inst_count[sphere_inst_count > 0])
        return t1


    #only for datasets with no categorical features
    def DBC(self,distance_func="default"):
        '''
        Calculate the DBC complexity measure defined in [1].

        -------
        Parameters:
        distance_func (string): The distance metric to calculate the distance between all hypersphere centers. 
        For now this value can only be "HEOM", since this is the only distance metric implemented.
        -------
        Returns:
        dbc_measure (float): the DBC complexity measure
        -------
        References:

        [1] Van der Walt CM, et al. (2008) Data measures that characterise
        classification problems. PhD thesis, University of Pretoria

        '''

        #find the hypersphere centers
        sphere_inst_count,radius=self.__get_sphere_count()
        inx=np.where(sphere_inst_count!=0)
        new_X = self.X[inx]
        new_y = self.y[inx]

        #calculate the distance between the hypersphere centers.
        new_dist_matrix,_ =self.__calculate_distance_matrix(new_X,distance_func=distance_func)

        #calculate the MST of using the hypersphere centers and find the number of vertixes linked by an edge in the MST that have a different class label
        n_inter = self.__calculate_n_inter(dist_matrix=new_dist_matrix,y=new_y)
      
        dbc_measure = n_inter/len(sphere_inst_count[sphere_inst_count > 0])

        
        return dbc_measure

    
    
    
    
    def LSC(self):
        '''
        Calculate the LSC measure defined in [1].

        ------
        Returns:
        lsc_measure (float): The value of the lsc complexity measure
        ------
        References:

        [1] Leyva E, González A, Perez R (2014) A set of complexity measures
        designed for applying meta-learning to instance selection. IEEE Transactions
        on Knowledge and Data Engineering 27(2):354-367
        '''


        #find the nearest neightbour of the oposite class for every sample
        nearest_enemy_inx,nearest_enemy_dist = self.__find_nearest_oposite_class_all()
        #print(nearest_enemy_dist)
        ls_count = []
        
        #for each sample count the amount of samples inside the hypersphere centered at the sample with radius equal
        #to the distance to the nearest sample of the oposite class.
        for i in range(len(self.dist_matrix)):
            count = 0
            for j in range(len(self.dist_matrix[i])):
                if(self.y[i]==self.y[j] and self.dist_matrix[i][j]<nearest_enemy_dist[i]):
                    count += 1
            ls_count.append(count)

        lsc_measure = 1-sum(ls_count)/(len(self.X)**2)
        return lsc_measure


    def Clust(self):
        '''
        Calculate the Clust complexity measure defined in [1].

        ------
        Returns:
        clust_measure (float): the value of the Clust complexity measure

        ------
        References:

        [1] Leyva E, González A, Perez R (2014) A set of complexity measures
        designed for applying meta-learning to instance selection. IEEE Transactions
        on Knowledge and Data Engineering 27(2):354-367
        '''
        nearest_enemy_inx,nearest_enemy_dist = self.__find_nearest_oposite_class_all()

        ls_count = []
        for i in range(len(self.class_count)):
            ls_count.append([])


        #for each sample calculate its local set
        for i in range(len(self.dist_matrix)):
            count = 0
            inxs = []
            for j in range(len(self.dist_matrix[i])):
                if(i!=j and self.y[i]==self.y[j] and self.dist_matrix[i][j]<nearest_enemy_dist[i]):
                    count += 1
                    inxs.append(j)
            cls_inx = np.where( self.classes == self.y[i])[0][0]
            #print(cls_inx)
            ls_count[cls_inx].append((count,i,inxs))
        
     
        
        core_count=0
        for i in range(len(ls_count)):
            clusters = []
            #sort all the classes LS
            ls_count[i].sort(key=itemgetter(0), reverse=True)

            
            for i2 in range(len(ls_count[i])):
                inCluster=False
                for c in clusters:
                    #if this instance is in the cluster core local set
                    clusterCoreLocalSet = ls_count[i][c[0]][2]
                    if (ls_count[i][i2][1] in clusterCoreLocalSet):
                        c[2].append(ls_count[i][i2][1])
                        inCluster=True
                #if it is not in the cluster core local set
                if(not inCluster):
                    clusterCoreInx = ls_count[i].index(ls_count[i][i2])
                    clusterCore = ls_count[i][i2]
                    clusterMembers = [clusterCore]
                    
                    cluster = [clusterCoreInx,clusterCore,clusterMembers]
                    clusters.append(cluster)
            #sum the number of cores
            core_count+=len(clusters)    
            

        
        #print(core_count)

        clust_measure = core_count/len(self.X)
        return clust_measure

    #only for datasets with no categorical features
    def NSG(self):
        '''
        Calculate the NSG complexity measure defined in [1].

        -------
        Returns: 
        nsg_measure (float): The value of the NSG complexity measure
        -------
        References:

        [1] Van der Walt CM, Barnard E (2007) Measures for the characterisation
        of pattern-recognition data sets. 18th Annual Symposium of the Pattern
        Recognition Association of South Africa
        '''
        sphere_inst_count,radius=self.__get_sphere_count()


        nsg_measure = sum(sphere_inst_count)/len(sphere_inst_count[sphere_inst_count > 0])
        return nsg_measure
    
    #only for datasets with no categorical features
    def ICSV(self):
        
        '''
        Calculate the ICSV complexity measure defined in [1].

        -------
        Returns: 
        icsv_measure (float): The value of the ICSV complexity measure
        -------
        References:

        [1] Van der Walt CM, Barnard E (2007) Measures for the characterisation
        of pattern-recognition data sets. 18th Annual Symposium of the Pattern
        Recognition Association of South Africa
        
        '''
        
        #find the hyperspheres and their radius
        sphere_inst_count,radius=self.__get_sphere_count()
        #print(radius)
        #print(len(sphere_inst_count))
        #print(len(sphere_inst_count))
        

        #calculate the density of each hypersphere
        n = len(self.class_inxs)
        density = []
        for i in range(len(radius)):
            #volume = 4/3 * math.pi* radius[i] **3
            volume = (math.pi**(n/2)/math.gamma(n/2 + 1)) * radius[i]**n
            density.append(sphere_inst_count[i]/volume)
            
        
        icsv_measure = 0


        mean = sum(density)/len(sphere_inst_count[sphere_inst_count > 0])
        #print(len(sphere_inst_count[sphere_inst_count > 0]))
        
        #calculate the icsv measure using the mean and densities of the spheres
        for i in range(len(radius)):
           if(density[i]!=0):
                icsv_measure+=(density[i]-mean)**2
        


        icsv_measure = icsv_measure/len(sphere_inst_count[sphere_inst_count > 0])
        icsv_measure = math.sqrt(icsv_measure)

        return icsv_measure

    def __calculate_cells(self,resolution,transpose_X,get_labels=0):
        '''
        Called py the purity and neighbourhood_separability functions.
        Creates a dictionary that maps each of the cells to the samples inside it. or if get_lables=1 it
        maps each of the cells to the class labels of each of the samples inside it.
        ------
        Parameters:
        resolution (int): an integer that determines in how many hypercubes the feature space will be divided into. More specificaly the number
        of cells per feature axis. If it equals 0 there is no particioning.
        transpose_X (numpy.array): the transpose of the X matrix containing the dataset saples
        get_labels: if 0 the reverse dictionary maps each cell to the index of the samples in X, if 1 the reverse dictionary maps each cell to the label of the samples.
        ------
        Returns:
        reverse_dic (dictionary): a dictionary that maps the cells to the samples or their labels.
        '''
        feature_bounds=[]
        steps = []
        #for every feature
        for j in range(len(self.X[0])):
            #(max of the feature - min of the feature)/(num of intervals +1)
            min_feature = min(transpose_X[j])
            max_feature = max(transpose_X[j])

          
            
           

              
            step = (max_feature-min_feature)/(resolution+1)
            steps.append(step)
            #calculate the hypercube bounds for each dimension
            
            feature_bounds.append(np.arange(min_feature,max_feature,step))
        

        #print(feature_bounds)
       
        sample_dic = {}
        #map each cell to the cell it belongs to (sample->cell)
        for s in range(len(self.X)):
            sample = self.X[s]
            for j in range(len(self.X[0])):
                for k in range(len(feature_bounds[j])):
                
                    if(sample[j]>=feature_bounds[j][k] and sample[j]<=feature_bounds[j][k]+steps[j]):
                        if(str(s) not in sample_dic):
                            sample_dic[str(s)]=""+str(k)
                        else:
                            sample_dic[str(s)]+="-"+str(k)
                        break
        
        
        
        '''
        for f in feature_bounds[1]:
            plt.axhline(y=f, color='r', linestyle='-')

        for f in feature_bounds[0]:
            plt.axvline(x=f, color='r', linestyle='-')

        
        for inx in self.class_inxs:
            plt.plot(self.X[inx,0],self.X[inx,1],"o",markersize=4)
        plt.show()
        '''

        #reverse the mapping (cell->sample)
        reverse_dic = {}
        for k,v in sample_dic.items():
            reverse_dic[v]= reverse_dic.get(v,[])
            #values are the class lables 
            if(get_labels==1):
                reverse_dic[v].append(self.y[int(k)]) 
            else:
                reverse_dic[v].append(int(k)) 

        
        return reverse_dic


    #only for datasets with no categorical features
    def purity(self,max_resolution=32):
        '''
        Calculates the purity complexity measure defined in [1].

        -----
        Parameters:
        max_resolution (int): the maximum resolution to consider to divide the feature space. The function will iterate from 0 (no partitioning) to max_resolution.
        -----
        Returns: 
        auc (float): Corresponds to the value of the purity complexity measure.
        -----
        References:

        [1] Singh S (2003) Prism{a novel framework for pattern recognition. Pattern
        Analysis & Applications 6(2):134-149

        '''
        transpose_X = np.transpose(self.X)
        purities = [] 
        #multiple resolutions
        for i in range(max_resolution):
            reverse_dic=self.__calculate_cells(i,transpose_X,get_labels=1)
            purity = 0
            #calculate purity
            for cell in reverse_dic:
                
                classes = self.classes
                class_counts = [0]*len(classes)
                num_classes = len(classes)

                #calculate the class counts on each cell
                for label in reverse_dic[cell]:
                    class_counts[np.where(classes==label)[0][0]]+=1
                class_sum=0
                #print(class_counts)
                #sum the values on each cell
                for count in class_counts:
                    class_sum+=((count/sum(class_counts))-(1/num_classes))**2
                    
               

                cell_purity = math.sqrt((num_classes/(num_classes-1))*class_sum)
                
                
                purity+=cell_purity*(sum(class_counts))/len(self.X)
            purities.append(purity)
        
        w_purities = []
        for i in range(len(purities)):
            new_p = purities[i]*(1/2**(i))
            w_purities.append(new_p)
        
        norm_resolutions = [x/max_resolution for x in list(range(max_resolution))]
        norm_purities = [(x-min(w_purities))/(max(w_purities)-min(w_purities)) for x in w_purities]
        print(norm_purities)
        auc=sklearn.metrics.auc(norm_resolutions,norm_purities)
        return auc/0.702
    

    #only for datasets with no categorical features
    def neighbourhood_separability(self,max_resolution=32):
        '''
        Calculates the neighbourhood_separability complexity measure defined in [1].

        -----
        Parameters:
        max_resolution (int): the maximum resolution to consider to divide the feature space. The function will iterate from 0 (no partitioning) to max_resolution.
        -----
        Returns: 
        final_auc (float): Corresponds to the value of the neighbourhood_separability complexity measure.
        -----
        References:
        
        [1] Singh S (2003) Prism{a novel framework for pattern recognition. Pattern
        Analysis & Applications 6(2):134-149

        '''
        transpose_X = np.transpose(self.X)
        neigh_sep = []
        #multiple resolutions
        for i in range(max_resolution):
            #print("Resolution: "+str(i))
            reverse_dic=self.__calculate_cells(i,transpose_X)
            reverse_dic_labels=self.__calculate_cells(i,transpose_X,get_labels=1)
            average_ns=0
            for cell in reverse_dic:
                
                average_auc=0
                for sample in reverse_dic[cell]:
                    props = []
                    same_class_num = len(np.where(reverse_dic_labels[cell] == self.y[sample])[0])
                    
                    #according to the original paper limiting the max num to 11 is good practice since for large datasets the process
                    #becomes too costly
                    if(same_class_num>11):
                        same_class_num=11

                    #does it still count itself?
                    for k in range(same_class_num):
                        #checks all neighbours and not just the ones the cell
                        count=self.__knn(sample,copy.copy(self.dist_matrix[sample]),k+1)
                        cls_inx = np.where( self.classes == self.y[sample])[0][0]
                        class_count=count[cls_inx]
                        #print(count)
                        prop = class_count/sum(count)
                        props.append(prop)
                    

                    #if the sample is the only one of its class in the cell
                    if(len(props)<2):
                        
                        auc=0
                    else:
                        norm_k = [x/same_class_num for x in list(range(same_class_num))]
                        auc = sklearn.metrics.auc(norm_k, props)
                       
                    average_auc+=auc
                average_auc/=len(reverse_dic[cell])
                average_ns+=average_auc*(len(reverse_dic[cell])/len(self.X))
            neigh_sep.append(average_ns)
        w_neigh_sep = []
        for i in range(len(neigh_sep)):
            w_neigh_sep.append(neigh_sep[i]*(1/2**i))
        
        norm_resolutions = [x/max_resolution for x in list(range(max_resolution))]
        final_auc=sklearn.metrics.auc(norm_resolutions,w_neigh_sep)
        return final_auc


    #only for datasets with no categorical features
    def F1(self):
        '''
        Calculates the F1 measure defined in [1]. 
        Uses One vs One method to handle multiclass datasets.
        -----
        Returns:
        f1_val (array): The value of f1 measure for each feature.
        -----
        References:

        [1] Luengo J, Fernández A, García S, Herrera F (2011) Addressing data
        complexity for imbalanced data sets: analysis of smote-based oversampling and
        evolutionary undersampling. Soft Computing 15(10):1909-1936

        '''
        f1s=[]

        #one vs one method
        for i in range(len(self.class_inxs)):
            for j in range(i+1,len(self.class_inxs)):
                
                #get the samples of the 2 classes being considered this iteration
                sample_c1 = self.X[self.class_inxs[i]]
                sample_c2 = self.X[self.class_inxs[j]]

                avg_c1 = np.mean(sample_c1,0)
                avg_c2 = np.mean(sample_c2,0)


                std_c1 = np.std(sample_c1,0)
                std_c2 = np.std(sample_c2,0)
                

                
                f1 = ((avg_c1-avg_c2)**2)/(std_c1**2+std_c2**2)
                
                
                f1[np.isinf(f1)]=0
                f1[np.isnan(f1)]=0

                f1 = 1/(1+f1)
                f1s.append(f1)

        #averge the all the values obtained with the one vs one comparison
        f1_val = np.mean(f1s,axis=0)
        return f1_val


    #only for datasets with no categorical features
    def F1v(self):
        '''
        Calculates the F1v measure defined in [1]. 
        Uses One vs One method to handle multiclass datasets.
        -----
        Returns:
        f1vs (array): The value of f1v measure for each one vs one combination of classes.
        -----
        References:

        [1] Lorena AC, Garcia LP, Lehmann J, Souto MC, Ho TK (2019) How com-
        plex is your classification problem? a survey on measuring classification
        complexity. ACM Computing Surveys (CSUR) 52(5):1-34
                
        '''
        f1vs=[]
        #one vs one method
        for i in range(len(self.class_inxs)):
            for j in range(i+1,len(self.class_inxs)):
                sample_c1 = self.X[self.class_inxs[i]]
                sample_c2 = self.X[self.class_inxs[j]]

                avg_c1 = np.mean(sample_c1,0)
                avg_c2 = np.mean(sample_c2,0)
                W =  (len(sample_c1)*np.cov(sample_c1, rowvar=False, ddof=1) + len(sample_c2)*np.cov(sample_c2, rowvar=False, ddof=1))/(len(sample_c1)+len(sample_c2))
                B = np.outer(avg_c1-avg_c2, avg_c1-avg_c2)
                d = np.matmul(scipy.linalg.pinv(W), avg_c1-avg_c2)
                
            
            
                f1v=(np.matmul(d.T,np.matmul(B,d)))/(np.matmul(d.T,np.matmul(W,d)))

                f1v = 1/(1+f1v)
                f1vs.append(f1v)
        return f1vs


    #only for datasets with no categorical features
    def F2(self):
        '''
        Calculates the F2 measure defined in [1]. 
        Uses One vs One method to handle multiclass datasets.
        -----
        Returns:
        f2s (array): The value of f2 measure for each one vs one combination of classes.
        -----
        References:

        [1] Lorena AC, Garcia LP, Lehmann J, Souto MC, Ho TK (2019) How com-
        plex is your classification problem? a survey on measuring classification
        complexity. ACM Computing Surveys (CSUR) 52(5):1-34
        '''
        f2s=[]
        #one vs one method
        for i in range(len(self.class_inxs)):
            for j in range(i+1,len(self.class_inxs)):
                sample_c1 = self.X[self.class_inxs[i]]
                sample_c2 = self.X[self.class_inxs[j]]

                
                maxmax = np.max([np.max(sample_c1,axis=0),np.max(sample_c2,axis=0)],axis=0)
                maxmin = np.max([np.min(sample_c1,axis=0),np.min(sample_c2,axis=0)],axis=0)
                minmin = np.min([np.min(sample_c1,axis=0),np.min(sample_c2,axis=0)],axis=0)
                minmax = np.min([np.max(sample_c1,axis=0),np.max(sample_c2,axis=0)],axis=0)

                numer=np.maximum(0.0, minmax - maxmin) 
            
                denom=(maxmax - minmin)


                n_d = numer/denom
               
                n_d[np.isinf(n_d)]=0
                n_d[np.isnan(n_d)]=0
                
                f2 = np.prod(n_d)
                f2s.append(f2)

        return f2s


    #only for datasets with no categorical features
    def F3(self):
        '''
        Calculates the F3 measure defined in [1]. 
        Uses One vs One method to handle multiclass datasets.
        -----
        Returns:
        f3s (array): The value of f3 measure for each one vs one combination of classes.
        -----
        References:

        [1] Lorena AC, Garcia LP, Lehmann J, Souto MC, Ho TK (2019) How com-
        plex is your classification problem? a survey on measuring classification
        complexity. ACM Computing Surveys (CSUR) 52(5):1-34
        '''
        f3s=[]
        #one vs one method
        for i in range(len(self.class_inxs)):
            for j in range(i+1,len(self.class_inxs)):
                sample_c1 = self.X[self.class_inxs[i]]
                sample_c2 = self.X[self.class_inxs[j]]

                maxmin = np.max([np.min(sample_c1,axis=0),np.min(sample_c2,axis=0)],axis=0)
                minmax = np.min([np.max(sample_c1,axis=0),np.max(sample_c2,axis=0)],axis=0)

                transpose_X = np.transpose(self.X)

                overlap_count = []

                #for every feature
                for k in  range(len(transpose_X)):
                    feature = transpose_X[k]
                    count = 0

                    #for every sample
                    for value in feature:
                        #check if the sample is inside the bounds
                        if(value>=maxmin[k] and value<=minmax[k]):
                            count+=1
                    overlap_count.append(count)
            

                min_overlap = min(overlap_count)
                f3 = min_overlap/(len(self.X[self.class_inxs[i]])+len(self.X[self.class_inxs[j]]))
                f3s.append(f3)
        return f3s
    

    def F3_imb(self):
        f3s=[]


        #one vs one method
        for i in range(len(self.class_inxs)):
            for j in range(i+1,len(self.class_inxs)):
                sample_c1 = self.X[self.class_inxs[i]]
                sample_c2 = self.X[self.class_inxs[j]]

                maxmin = np.max([np.min(sample_c1,axis=0),np.min(sample_c2,axis=0)],axis=0)
                minmax = np.min([np.max(sample_c1,axis=0),np.max(sample_c2,axis=0)],axis=0)

                transpose_X = np.transpose(self.X)

                overlap_count = []

                #for every feature
                for k in  range(len(transpose_X)):
                    feature = transpose_X[k]
                    count = 0

                    #for every sample
                    for value in feature:
                        #check if the sample is inside the bounds
                        if(value>=maxmin[k] and value<=minmax[k]):
                            count+=1
                    overlap_count.append(count)
            

                min_overlap = min(overlap_count)
                f3 = min_overlap/(len(self.X[self.class_inxs[i]])+len(self.X[self.class_inxs[j]]))
                f3s.append(f3)
        return f3s
    #only for datasets with no categorical features
    def F4(self):
        '''
        Calculates the F4 measure defined in [1]. 
        Uses One vs One method to handle multiclass datasets.
        -----
        Returns:
        f4s (array): The value of f4 measure for each one vs one combination of classes.
        -----
        References:

        [1] Lorena AC, Garcia LP, Lehmann J, Souto MC, Ho TK (2019) How com-
        plex is your classification problem? a survey on measuring classification
        complexity. ACM Computing Surveys (CSUR) 52(5):1-34
        '''

       
        f4s=[]
        #one vs one method
        for i2 in range(len(self.class_inxs)):
            for j2 in range(i2+1,len(self.class_inxs)):
                
                
                #create the subset with just the 2 classes
                valid_inxs_c1 = self.class_inxs[i2]
                valid_inxs_c2 = self.class_inxs[j2]
                sample_c1 = self.X[valid_inxs_c1]
                sample_c2 = self.X[valid_inxs_c2]
                sample_c1_y = self.y[valid_inxs_c1]
                sample_c2_y = self.y[valid_inxs_c2]

                X = np.concatenate((sample_c1,sample_c2),axis=0)
                y = np.concatenate((sample_c1_y,sample_c2_y),axis=0)
                valid_inxs_c1 = np.where(y==self.classes[i2])[0]
                valid_inxs_c2 = np.where(y==self.classes[j2])[0]
                
                
                transpose_X = np.transpose(X)

                #while there are still samples in X
                while len(X)>0:

                    #check if one of the classes is not represented
                    if(len(sample_c1)==0 or len(sample_c2)==0):
                        maxmin = np.full(len(X[1]),np.inf)
                        minmax = np.full(len(X[1]),-np.inf)
                    else:
                        maxmin = np.max([np.min(sample_c1,axis=0),np.min(sample_c2,axis=0)],axis=0)
                        minmax = np.min([np.max(sample_c1,axis=0),np.max(sample_c2,axis=0)],axis=0)

                    overlap_count = []
                    inx_lists = []


                    #for each feature
                    for i in  range(len(transpose_X)):
                        feature = transpose_X[i]
                        inx_list = [] 
                        count = 0
                        #for each sample
                        for j in range(len(feature)):
                            value = feature[j]
                            if(value>=maxmin[i] and value<=minmax[i]):
                                count+=1
                                inx_list.append(j)
                        
                        overlap_count.append(count)
                        inx_lists.append(inx_list)
                        
                    
                    #determine the feature with the least overlap
                    min_overlap = min(overlap_count)
                    min_inx = overlap_count.index(min_overlap)
                    min_overlap_inx = inx_lists[min_inx]


                    #remove that feature
                    valid_features = list(range(0,min_inx)) + list(range(min_inx+1,len(transpose_X)))

                    if(len(min_overlap_inx)==0 or len(valid_features)==0):
                        break


                    
                    

                    new_X = []
                    new_y = []
                    


                    #create the new values for X and y after removing the feature and remove the samples that can correctly classified with just that feature
                    for inx in range(len(X)):
                       
                        if inx in min_overlap_inx:
                            sample = []
                            for ft in valid_features:
                               
                                sample.append(X[inx,ft])
                            new_X.append(sample)
                            new_y.append(y[inx])
                    X = np.array(new_X)
                    y = np.array(new_y)
                    transpose_X = np.transpose(X)
                    
                    valid_inxs_c1 = []
                    sample_c1 = []
                    for inx in range(len(y)):
                        if y[inx]==self.classes[i2]:
                            valid_inxs_c1.append(inx)
                            sample_c1.append(X[inx])
                    valid_inxs_c2 = []
                    sample_c2 = []
                    for inx in range(len(y)):
                        if y[inx]==self.classes[j2]:
                            valid_inxs_c2.append(inx)
                            sample_c2.append(X[inx])

                    sample_c1 = np.array(sample_c1)
                    sample_c2 = np.array(sample_c2)
                    #print(sample_c1)
                    #print(sample_c2)
                f4=len(min_overlap_inx)/(len(sample_c1_y)+len(sample_c2_y))
                f4s.append(f4)
        return f4s


    
    def __class_overlap(self,class_samples,other_samples):
        '''
        Called by the input_noise function.
        Calculates the class overlap between two classes.
        -------
        Parameters:
        class_samples (array): Array with the samples of the first class.
        other_sample (array): Array with the sample of the second class.
        -------
        Returns:
        count (int): the number of samples from the first class that are in the domain of the second class.
        '''
        min_class = np.min(other_samples,axis=0)
        max_class = np.max(other_samples,axis=0)
        count = 0
        for sample in class_samples:
            for i in range(len(sample)):
                feature = sample[i]
                if(feature>min_class[i] and feature<max_class[i]):
                   count+=1 
                
        return count


    #only for datasets with no categorical features
    def input_noise(self):
        '''
        Calculate the input noise metric defined in [1].
        Uses one vs one method if the dataset contains more than 2 classes.
        -----
        Returns:
        ins (array): the input noise value for each each one vs one combination of classes.
        -----
        References:

        [1] Van der Walt CM, Barnard E (2007) Measures for the characterisation
        of pattern-recognition data sets. 18th Annual Symposium of the Pattern
        Recognition Association of South Africa
        '''
        ins = []
        for i in range(len(self.class_inxs)):
            for j in range(i+1,len(self.class_inxs)):
                X = self.X
                valid_inxs_c1 = self.class_inxs[i]
                valid_inxs_c2 = self.class_inxs[j]
                sample_c1 = X[valid_inxs_c1]
                sample_c2 = X[valid_inxs_c2]
                
                total_count=0
                total_count+=self.__class_overlap(sample_c1,sample_c2)
                total_count+=self.__class_overlap(sample_c2,sample_c1)

                ins.append(total_count/(len(X)*len(X[0])))
        return ins

    
    def borderline(self,k=5):
        '''
        Calculates the borderline examples metric defined in [1].

        ------
        Parameters:
        k (int): The number of nearest neighbours to consider.
        ------
        Returns:
        borderline (float): the percentage of borderline examples in the dataset.
        ------
        References:

        [1] Napiera la K, Stefanowski J, Wilk S (2010) Learning from imbalanced data
        in presence of noisy and borderline examples. In: International Conference
        on Rough Sets and Current Trends in Computing, Springer, pp 158-167
        
        '''
        borderline_count=0
        for i in range(len(self.X)):
            count=self.__knn(i,copy.copy(self.dist_matrix[i]),k)
            cls_inx = np.where( self.classes == self.y[i])[0][0]
            #check this
            if((sum(count)-count[cls_inx])>=2):
                borderline_count+=1
        borderline = borderline_count*100/len(self.X)
        return borderline

    def deg_overlap(self,k=5):
        '''
        Calculates the degree of overlap metric defined in [1].

        ------
        Parameters:
        k (int): The number of nearest neighbours to consider.
        ------
        Returns:
        deg_ov (float): The degree of overlap complexity measure.
        ------
        References:

        [1] Mercier M, Santos M, Abreu P, Soares C, Soares J, Santos J (2018)
        Analysing the footprint of classifiers in overlapped and imbalanced con-
        texts. In: International Symposium on Intelligent Data Analysis, Springer,
        pp 200-212
        '''
        deg=0
        for i in range(len(self.X)):
            count=self.__knn(i,copy.copy(self.dist_matrix[i]),k)
            cls_inx = np.where( self.classes == self.y[i])[0][0]
            if(count[cls_inx]!=k):
                deg+=1
        deg_ov = deg/len(self.X)
        return deg_ov

    # -*- coding: utf-8 -*-


    def l1(self):

 
        scaler = sklearn.preprocessing.StandardScaler()
        svc = sklearn.svm.LinearSVC(penalty="l2",loss="hinge",C=2.0,tol=10e-3,max_iter=1000,random_state=0)
        
        svc_pipeline = sklearn.pipeline.Pipeline([("scaler", scaler), ("svc", svc)])
        sum_err_dist = []

        for i in range(len(self.class_inxs)):
            for j in range(i+1,len(self.class_inxs)):
    
                valid_inxs_c1 = self.class_inxs[i]
                valid_inxs_c2 = self.class_inxs[j]
                sample_c1 = self.X[valid_inxs_c1]
                sample_c2 = self.X[valid_inxs_c2]

                l_c1 = self.y[valid_inxs_c1]
                l_c2 = self.y[valid_inxs_c2]

                sub_X = np.concatenate((sample_c1,sample_c2))
                sub_y = np.concatenate((l_c1,l_c2))

                svc_pipeline.fit(sub_X,sub_y)

                y_pred = svc_pipeline.predict(sub_X)
                misclassified_insts = sub_X[y_pred != sub_y, :]

                if misclassified_insts.size:
                    insts_dists = svc_pipeline.decision_function(misclassified_insts)
                else:
                    insts_dists = np.array([0.0], dtype=float)

                sum_err_dist.append(np.linalg.norm(insts_dists, ord=1) / (len(sub_y)))

        sum_err_dist = np.array(sum_err_dist)
        l1 = 1.0 - 1.0 / (1.0 + sum_err_dist)
        return l1

    def l2(self):
        print("bruh")
        scaler = sklearn.preprocessing.StandardScaler()
        svc = sklearn.svm.LinearSVC(penalty="l2",loss="hinge",C=2.0,tol=10e-3,max_iter=1000,random_state=0)
        
        svc_pipeline = sklearn.pipeline.Pipeline([("scaler", scaler), ("svc", svc)])
        l2 = []

        for i in range(len(self.class_inxs)):
            for j in range(i+1,len(self.class_inxs)):
    
                valid_inxs_c1 = self.class_inxs[i]
                valid_inxs_c2 = self.class_inxs[j]
                sample_c1 = self.X[valid_inxs_c1]
                sample_c2 = self.X[valid_inxs_c2]

                l_c1 = self.y[valid_inxs_c1]
                l_c2 = self.y[valid_inxs_c2]

                sub_X = np.concatenate((sample_c1,sample_c2))
                sub_y = np.concatenate((l_c1,l_c2))

                svc_pipeline.fit(sub_X,sub_y)

                y_pred = svc_pipeline.predict(sub_X)

                error = sklearn.metrics.zero_one_loss(y_true=sub_y, y_pred=y_pred, normalize=True)

                l2.append(error)
        
        return l2

    def l3(self):

        scaler = sklearn.preprocessing.StandardScaler()
        svc = sklearn.svm.LinearSVC(penalty="l2",loss="hinge",C=2.0,tol=10e-3,max_iter=1000,random_state=0)
        
        svc_pipeline = sklearn.pipeline.Pipeline([("scaler", scaler), ("svc", svc)])
        l3 = []


        
        for i in range(len(self.class_inxs)):
            for j in range(i+1,len(self.class_inxs)):
    
                valid_inxs_c1 = self.class_inxs[i]
                valid_inxs_c2 = self.class_inxs[j]
                sample_c1 = self.X[valid_inxs_c1]
                sample_c2 = self.X[valid_inxs_c2]

                l_c1 = self.y[valid_inxs_c1]
                l_c2 = self.y[valid_inxs_c2]

                sub_X = np.concatenate((sample_c1,sample_c2))
                sub_y = np.concatenate((l_c1,l_c2))
                
                svc_pipeline.fit(sub_X, sub_y)
                
                
                valid_inxs = []
                valid_inxs.append(valid_inxs_c1)
                valid_inxs.append(valid_inxs_c2)
               
                X_interp, y_interp=self.__interpolate_samples(class_inxs=valid_inxs)
          

                y_pred = svc_pipeline.predict(X_interp)

                error = sklearn.metrics.zero_one_loss(y_true=y_interp, y_pred=y_pred, normalize=True)

                l3.append(error)


        return l3





    def ONB_avg(self,dist_id = "manhattan"):

        """

        ONB complexity metric differentiating (averaged) by class

        dataset is the dataframe of the dataset that will be evaluated, where the last column is the class

        dist_id is the type of distance metric that will be utilised (with "manhattan" as the default metric, but "euclidean" was also used in the study)

        """
        featu = pd.DataFrame(self.X) #feature part of the dataframe
        
        clas = pd.DataFrame(self.y) #class part of the dataframe
        
        clas.rename(columns = {0 : 'class'}, inplace = True)
        
        dataset = featu.join(clas)

        clas_dif = np.unique(clas) #array of the different clases in the dataset   

        dist= 0 #DistanceMetric.get_metric(dist_id) #indicating which distance metric will be used    

        dtf_dist =  pd.DataFrame(dist.pairwise(featu)) #distance matrix as a dataframe    

        lista_el_cla = [] #empty list for the elements of each class

        el_cla = [] #empty list for the number of elements in each class

        for cla in clas_dif:

            lista_el_cla = lista_el_cla + [list(dataset.loc[dataset["class"]==cla].index.values)] #add the elements of each class

            el_cla = el_cla + [len(list(dataset.loc[dataset["class"]==cla].index.values))] #add the number of elements in each class

        cl=0 #counter for the classes, starting at 0

        avg=0 #variable where the ratios of balls per number of elements of each class will be added, and later averaged

        for cla in clas_dif: #for each class

            bolascla=0 #counter for the number of balls necessary for class coverage 

            falta = lista_el_cla[cl] #list of uncovered instances of that class, which initially includes all elements of said class

            while len(falta)!=0: #while there are uncovered elements of said class

                bolaslist=[] #empty list for the elements of the ball that covers the most instances for this iteration

                for j in falta: #for each uncovered element

                    mininter = min(dtf_dist.loc[j,dataset["class"]!=cla]) #the biggest possible radius that does not cover elements from other classes is computed

                    s=dtf_dist.iloc[j, falta]<mininter #the instances included in that ball are obtained

                    bolas = [falta.index(i) for i in s[s].index.values] #said instances are saved as a list

                    if len(bolas) > len(bolaslist): #if this ball covers more instances than the previous best candidate

                        bolaslist = bolas #its covered elements are saved

                bolascla+=1 #the total of chosen balls for the coverage of said class increases by 1

                for ele in sorted(bolaslist, reverse = True): 

                    del falta[ele] #the elements covered by chosen ball are deleted from the list of uncovered instances 

            avg += (bolascla / el_cla[cl]) #the ratio between the number of balls necessary to cover the points of a class and the number of points of that class is added to the sum

            cl+=1 #continue to the next class

        return avg / len(clas_dif) #return the average of the ratios between the number of balls necessary to cover the points of a class and the number of points of that class





    def ONB_tot(self, dist_id = "manhattan"):

        """

        ONB complexity metric using the total number of balls

        dataset is the dataframe of the dataset that will be evaluated, where the last column is the class

        dist_id is the type of distance metric that will be utilised (with "manhattan" as the default metric, but "euclidean" was also used in the study)

        """
        
        featu = pd.DataFrame(self.X) #feature part of the dataframe
        print(featu)
        clas = pd.DataFrame(self.y) #class part of the dataframe
        
        clas.rename(columns = {0 : 'class'}, inplace = True)
        print(clas)
        dataset = featu.join(clas)

        print(dataset)
        tam = dataset.shape[0] #number of instances of the dataframe

        clas_dif = np.unique(clas) #array of the different clases in the dataset   

        dist= 0 #DistanceMetric.get_metric(dist_id) #indicating which distance metric will be used     

        dtf_dist = pd.DataFrame(dist.pairwise(featu)) #distance matrix as a dataframe   

        lista_el_cla = [] #empty list for the elements of each class

        for cla in clas_dif:

            lista_el_cla = lista_el_cla + [list(dataset.loc[dataset["class"]==cla].index.values)] #add the elements of each class

        nbolas = 0 #counter for the balls necessary for the full coverage of the dataset, starting at 0

        cl=0 #counter for the classes, starting at 0

        for cla in clas_dif: #for each class

            falta = lista_el_cla[cl] #list of uncovered instances of that class, which initially includes all elements of said class

            while len(falta)!=0: #while there are uncovered elements of said class

                bolaslist=[] #empty list for the elements of the ball that covers the most instances for this iteration

                for j in falta: #for each uncovered element

                    mininter = min(dtf_dist.loc[j,dataset["class"]!=cla]) #the biggest possible radius that does not cover elements from other classes is computed

                    s=dtf_dist.iloc[j, falta]<mininter #the instances included in that ball are obtained

                    bolas = [falta.index(i) for i in s[s].index.values] #said instances are saved as a list

                    if len(bolas) > len(bolaslist): #if this ball covers more instances than the previous best candidate

                        bolaslist = bolas #its covered elements are saved

                nbolas+=1 #the total of chosen balls for the coverage increases by 1

                for ele in sorted(bolaslist, reverse = True): 

                    del falta[ele] #the elements covered by chosen ball are deleted from the list of uncovered instances 

            cl+=1 #continue to the next class  

        return nbolas / tam #return the ratio between the number of balls necessary for full coverage and the number of instances of the dataset


