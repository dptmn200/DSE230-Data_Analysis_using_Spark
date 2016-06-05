# coding: utf-8

# Name: Deepthi Mysore Nagaraj 
# Email: dmysoren@eng.ucsd.edu
# PID: A53110637

from pyspark import SparkContext
sc = SparkContext()

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

from string import split,strip

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel,RandomForest
from pyspark.mllib.util import MLUtils


# In[10]:

# Read the file into an RDD
path='/covtype/covtype.data'
inputRDD=sc.textFile(path)


# In[11]:

# Transform the text RDD into an RDD of LabeledPoints
Data=inputRDD.map(lambda line: [float(strip(x)) for x in line.split(',')]).map(lambda x: LabeledPoint(x[-1],x[0:-1])) ## Fillin##
Data.first()
        

# ### Making the problem binary
# 
# The implementation of BoostedGradientTrees in MLLib supports only binary problems. the `CovTYpe` problem has
# 7 classes. To make the problem binary we choose the `Lodgepole Pine` (label = 2.0). We therefor transform the dataset to a new dataset where the label is `1.0` is the class is `Lodgepole Pine` and is `0.0` otherwise.

# In[13]:

Label=2.0
default_val = {Label:1.0}
Data=inputRDD.map(lambda line: [float(x) for x in line.split(',')]).map(lambda V:LabeledPoint(default_val.setdefault(V[-1], 0.0), V[0:-1]))
    

# ### Reducing data size
# In order to see the effects of overfitting more clearly, we reduce the size of the data by a factor of 10

(trainingData,testData)=Data.randomSplit([0.7,0.3],seed=255)

## Gradient boosting 

from time import time
errors={}
for depth in [10]:
    start=time()
    ##FILLIN to generate 10 trees ##
    model=GradientBoostedTrees.trainClassifier(trainingData,categoricalFeaturesInfo={} ,numIterations=10,maxDepth = depth,learningRate = 0.4)
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
