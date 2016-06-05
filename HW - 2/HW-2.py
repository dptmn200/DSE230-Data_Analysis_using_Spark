# -*- coding: utf-8 -*- 
# Name: Deepthi Mysore Nagaraj
# Email: dmysoren@eng.ucsd.edu
# PID: A53110637

from pyspark import SparkContext
sc = SparkContext()

### Part 0 - Get data

def print_count(rdd):
    print 'Number of elements:', rdd.count()

with open('../Data/hw2-files-20gb.txt') as f:
    files = [l.strip() for l in f.readlines()]
rdd = sc.textFile(','.join(files)).cache()
print_count(rdd)


### Part 1

## 1)
import ujson

def safe_parse(raw_json):
    try:
        load_data = ujson.loads(raw_json)  
    except ValueError, e:
        return '0', 'NA'
    
    if 'id_str' in load_data and 'user' in load_data: 
        return load_data['user']['id_str'].encode('utf-8'), load_data['text'].encode('utf-8') 
    else:
        return '0', 'NA'

tweets = rdd.map(lambda x: safe_parse(x)).filter(lambda x: x[0] <> '0' or x[1] <> 'NA').cache()


## 2)

def print_users_count(count):
    print 'The number of unique users is:', count

# your code here
print_users_count(tweets.map(lambda x:x[0]).distinct().count())

### Part 2

## 1)
# your code here

import pickle

with open('../Data/users-partition.pickle') as f:
    partition_dict = pickle.load(f)

partition = sc.broadcast(partition_dict)


## 2)
# your code here

def partitionid(id_str):
    if partition.value.has_key(id_str): 
        return partition.value[id_str]
    else:
        return 7


sorted_output = tweets.map(lambda x: (partitionid(x[0]))).countByValue().items()

## 3)

def print_post_count(counts):
    for group_id, count in counts:
        print 'Group %d posted %d tweets' % (group_id, count)

print_post_count(sorted_output)


### Part 3

################################ Tokenizing code - start ##################################
# %load happyfuntokenizing.py
#!/usr/bin/env python

"""
This code implements a basic, Twitter-aware tokenizer.

A tokenizer is a function that splits a string of text into words. In
Python terms, we map string and unicode objects into lists of unicode
objects.

There is not a single right way to do tokenizing. The best method
depends on the application.  This tokenizer is designed to be flexible
and this easy to adapt to new domains and tasks.  The basic logic is
this:

1. The tuple regex_strings defines a list of regular expression
   strings.

2. The regex_strings strings are put, in order, into a compiled
   regular expression object called word_re.

3. The tokenization is done by word_re.findall(s), where s is the
   user-supplied string, inside the tokenize() method of the class
   Tokenizer.

4. When instantiating Tokenizer objects, there is a single option:
   preserve_case.  By default, it is set to True. If it is set to
   False, then the tokenizer will downcase everything except for
   emoticons.

The __main__ method illustrates by tokenizing a few examples.

I've also included a Tokenizer method tokenize_random_tweet(). If the
twitter library is installed (http://code.google.com/p/python-twitter/)
and Twitter is cooperating, then it should tokenize a random
English-language tweet.


Julaiti Alafate:
  I modified the regex strings to extract URLs in tweets.
"""

__author__ = "Christopher Potts"
__copyright__ = "Copyright 2011, Christopher Potts"
__credits__ = []
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License: http://creativecommons.org/licenses/by-nc-sa/3.0/"
__version__ = "1.0"
__maintainer__ = "Christopher Potts"
__email__ = "See the author's website"

######################################################################

import re
import htmlentitydefs

######################################################################
# The following strings are components in the regular expression
# that is used for tokenizing. It's important that phone_number
# appears first in the final regex (since it can contain whitespace).
# It also could matter that tags comes after emoticons, due to the
# possibility of having text like
#
#     <:| and some text >:)
#
# Most imporatantly, the final element should always be last, since it
# does a last ditch whitespace-based tokenization of whatever is left.

# This particular element is used in a couple ways, so we define it
# with a name:
emoticon_string = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""

# The components of the tokenizer:
regex_strings = (
    # Phone numbers:
    r"""
    (?:
      (?:            # (international)
        \+?[01]
        [\-\s.]*
      )?            
      (?:            # (area code)
        [\(]?
        \d{3}
        [\-\s.\)]*
      )?    
      \d{3}          # exchange
      [\-\s.]*   
      \d{4}          # base
    )"""
    ,
    # URLs:
    r"""http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"""
    ,
    # Emoticons:
    emoticon_string
    ,    
    # HTML tags:
     r"""<[^>]+>"""
    ,
    # Twitter username:
    r"""(?:@[\w_]+)"""
    ,
    # Twitter hashtags:
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"""
    ,
    # Remaining word types:
    r"""
    (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots. 
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """
    )

######################################################################
# This is the core tokenizing regex:
    
word_re = re.compile(r"""(%s)""" % "|".join(regex_strings), re.VERBOSE | re.I | re.UNICODE)

# The emoticon string gets its own regex so that we can preserve case for them as needed:
emoticon_re = re.compile(regex_strings[1], re.VERBOSE | re.I | re.UNICODE)

# These are for regularizing HTML entities to Unicode:
html_entity_digit_re = re.compile(r"&#\d+;")
html_entity_alpha_re = re.compile(r"&\w+;")
amp = "&amp;"

######################################################################

class Tokenizer:
    def __init__(self, preserve_case=False):
        self.preserve_case = preserve_case

    def tokenize(self, s):
        """
        Argument: s -- any string or unicode object
        Value: a tokenize list of strings; conatenating this list returns the original string if preserve_case=False
        """        
        # Try to ensure unicode:
        try:
            s = unicode(s)
        except UnicodeDecodeError:
            s = str(s).encode('string_escape')
            s = unicode(s)
        # Fix HTML character entitites:
        s = self.__html2unicode(s)
        # Tokenize:
        words = word_re.findall(s)
        # Possible alter the case, but avoid changing emoticons like :D into :d:
        if not self.preserve_case:            
            words = map((lambda x : x if emoticon_re.search(x) else x.lower()), words)
        return words

    def tokenize_random_tweet(self):
        """
        If the twitter library is installed and a twitter connection
        can be established, then tokenize a random tweet.
        """
        try:
            import twitter
        except ImportError:
            print "Apologies. The random tweet functionality requires the Python twitter library: http://code.google.com/p/python-twitter/"
        from random import shuffle
        api = twitter.Api()
        tweets = api.GetPublicTimeline()
        if tweets:
            for tweet in tweets:
                if tweet.user.lang == 'en':            
                    return self.tokenize(tweet.text)
        else:
            raise Exception("Apologies. I couldn't get Twitter to give me a public English-language tweet. Perhaps try again")

    def __html2unicode(self, s):
        """
        Internal metod that seeks to replace all the HTML entities in
        s with their corresponding unicode characters.
        """
        # First the digits:
        ents = set(html_entity_digit_re.findall(s))
        if len(ents) > 0:
            for ent in ents:
                entnum = ent[2:-1]
                try:
                    entnum = int(entnum)
                    s = s.replace(ent, unichr(entnum))	
                except:
                    pass
        # Now the alpha versions:
        ents = set(html_entity_alpha_re.findall(s))
        ents = filter((lambda x : x != amp), ents)
        for ent in ents:
            entname = ent[1:-1]
            try:            
                s = s.replace(ent, unichr(htmlentitydefs.name2codepoint[entname]))
            except:
                pass                    
            s = s.replace(amp, " and ")
        return s

################################ Tokenizing code - start ##################################

from math import log

tok = Tokenizer(preserve_case=False)

def get_rel_popularity(c_k, c_all):
    return log(1.0 * c_k / c_all) / log(2)

def print_tokens(tokens, gid = None):
    group_name = "overall"
    if gid is not None:
        group_name = "group %d" % gid
    print '=' * 5 + ' ' + group_name + ' ' + '=' * 5
    for t, n in tokens:
        print "%s\t%.4f" % (t, n)
    print


## 1)

# Creating a Tokenized RDD which can be re-used in the next step as well.
tokenized = tweets.flatMap(lambda x: [(y,x[0]) for y in tok.tokenize(x[1])]).distinct().cache()

# Aggregate using reduceByKey
tokenized2 =  tokenized.map(lambda x: (x[0],1)).reduceByKey(lambda x,y: x + y).cache()

print_count(tokenized2)


## 2)

# Filter for tokens which were mentioned by atleast 100 users
tokencount2 = tokenized2.filter(lambda x: x[1]>=100).cache()    
print_count(tokencount2)
print_tokens(tokencount2.top(20, lambda x:x[1]))

## 3)


# your code here

tokenized5 = tokenized.map(lambda x:((partitionid(x[1]),x[0]),1)).reduceByKey(lambda x,y:x+y)\
                      .map(lambda x: (x[0][0],x[0][1],x[1])).map(lambda x:(x[1],(x[2],x[0])))\
                      .join(tokencount2).map(lambda (x,y):(y[0][1],(x,get_rel_popularity(y[0][0],y[1]))))\
                      .aggregateByKey([], lambda x, v: x + [v], lambda x,y: sorted(x + y, key=lambda x: (-x[1],x[0]))[:10]).collect()

for i in sorted(tokenized5):
    print_tokens(i[1], i[0])

## 4)

users_support = [
    (2, "Bernie Sanders"),
    (5, "Ted Cruz"),
    (6, "Donald Trump")
]

for gid, candidate in users_support:
    print "Users from group %d are most likely to support %s." % (gid, candidate)

