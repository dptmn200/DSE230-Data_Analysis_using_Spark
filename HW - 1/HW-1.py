# Name: Deepthi Mysore Nagaraj
# Email: dmysoren@eng.ucsd.edu
# PID: A53110637
from pyspark import SparkContext
sc = SparkContext()

textRDD = sc.newAPIHadoopFile('/data/Moby-Dick.txt',
                             'org.apache.hadoop.mapreduce.lib.input.TextInputFormat',
                             'org.apache.hadoop.io.LongWritable',
                             'org.apache.hadoop.io.Text',
                             conf={'textinputformat.record.delimiter': "\r\n\r\n"}) \
           .map(lambda x: x[1])

sentences=textRDD.flatMap(lambda x: x.split(". ")).map(lambda x: x.encode('utf-8'))

## Remove punctuation and convert to lower case
## Removal of punctuation and \r\n
import re
sentences1 = sentences.map(lambda x: re.sub(r'\r\n', " ", x)).map(lambda x: x.lower()) 
sentences1 = sentences1.map(lambda x: re.sub("[^0-9a-zA-Z ]", " ", x))

## Filter out sentences whose length = 0
## Split each sentence into words
sentences2 = sentences1.filter(lambda x: len(x)!=0).map(lambda x: x.split(" ")).map(lambda ws: [w for w in ws if w and w != ''])

def calc_ngrams(sentence):
    output=[]
    for i in range(len(sentence)-n+1):
        output.append((tuple(sentence[i:i+n]),1))
    return output

def printOutput(n,freq_ngramRDD):
    top=freq_ngramRDD.take(5)
    print '\n============ %d most frequent %d-grams'%(5,n)
    print '\nindex\tcount\tngram'
    for i in range(5):
        print '%d.\t%d: \t"%s"'%(i+1,top[i][0],' '.join(top[i][1]))

for n in range(1,6):
    # Put your logic for generating the sorted n-gram RDD here and store it in freq_ngramRDD variable
    final_ngram = sentences2.flatMap(calc_ngrams)
    freq_ngramRDD = final_ngram.reduceByKey(lambda x,y: x+y).map(lambda x:(x[1],x[0])).sortByKey(False)
    printOutput(n,freq_ngramRDD)