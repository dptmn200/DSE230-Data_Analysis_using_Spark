
# coding: utf-8

# Name: Deepthi Mysore Nagaraj 
# Email: dmysoren@eng.ucsd.edu
# PID: A53110637

# In[1]:

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

from string import split,strip

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.tree import RandomForest, RandomForestModel

from pyspark.mllib.util import MLUtils


# In[2]:

## This piece of code is needed as I'm running the pyspark on virtual box
import pyspark
sc = pyspark.SparkContext()


# ### Higgs data set
# * **URL:** http://archive.ics.uci.edu/ml/datasets/HIGGS#  
# * **Abstract:** This is a classification problem to distinguish between a signal process which produces Higgs bosons and a background process which does not.
# 
# **Data Set Information:**  
# The data has been produced using Monte Carlo simulations. The first 21 features (columns 2-22) are kinematic properties measured by the particle detectors in the accelerator. The last seven features are functions of the first 21 features; these are high-level features derived by physicists to help discriminate between the two classes. There is an interest in using deep learning methods to obviate the need for physicists to manually develop such features. Benchmark results using Bayesian Decision Trees from a standard physics package and 5-layer neural networks are presented in the original paper. The last 500,000 examples are used as a test set.
# 
# 

path='/HIGGS/HIGGS.csv'
inputRDD=sc.textFile(path)


# In[10]:

# Transform the text RDD into an RDD of LabeledPoints
Data=inputRDD.map(lambda line: [float(strip(x)) for x in line.split(',')]).map(lambda x: LabeledPoint(x[0],x[1:])) ## Fillin##
Data.take(5)


# In[11]:

# Reduce the data size

Data1 = inputRDD.sample(False,0.1, seed=255).cache()
(trainingData,testData)=Data.randomSplit([0.7,0.3],seed=255)


# In[13]:

## Gradient Boosting

from time import time
errors={}
for depth in [10]:
    start=time()
    ##FILLIN to generate 10 trees ##
    model=GradientBoostedTrees.trainClassifier(trainingData,categoricalFeaturesInfo={} ,numIterations=10,maxDepth = depth,learningRate=0.6)
    ##trainingData, categoricalFeaturesInfo={}, numIterations=10, maxDepth=depth
    #print model.toDebugString()
    errors[depth]={}
    dataSets={'train':trainingData,'test':testData}
    for name in dataSets.keys():  # Calculate errors on train and test sets
        data=dataSets[name]
        Predicted=model.predict(data.map(lambda x: x.features))
        LabelsAndPredictions=data.map(lambda x: x.label).zip(Predicted) ### FILLIN ###
        Err = LabelsAndPredictions.filter(lambda (v,p):v != p).count()/float(data.count())
        errors[depth][name]=Err
    print depth,errors[depth]
# print errors

