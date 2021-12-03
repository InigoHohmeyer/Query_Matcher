import math

import nltk
import np as np
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import numpy as numpy
from nltk.tokenize import word_tokenize
from scipy import spatial
import stop_list
import string

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
# list of words that we don't need to use when making our vectors
stopwords = stop_list.closed_class_stop_words
# number of documents which contain certain words in the queries and in the abstracts
freq_abstracts = {}
freq_queries = {}
#  the number of queries and the number of abstracts
no_of_queries = 0
no_of_abstracts = 0
# converts the text files into strings with spaces replacing newlines
with open('cran.qry', 'r') as file:
    query_file = file.read().replace('\n', ' ')
with open('cran.all.1400', 'r') as file:
    abstract_file = file.read().replace('\n', ' ')

# tokenizes the words so it is converted
queries = word_tokenize(query_file)
abstract = word_tokenize(abstract_file)
# makes it so I doesn't get removed
#pos tags the words
pos_queries = nltk.pos_tag(queries)
pos_abstract = nltk.pos_tag(abstract)


j = 0
for i in queries:
    if i == ".I":
        queries[j] = "INDEX"

    j += 1

j = 0
for i in abstract:
    if i == ".I":
        abstract[j] = "INDEX"
    j += 1

# gets rid of all the punctuation, numbers and stopwords in both list
queries = [''.join(c for c in s if c not in string.punctuation) for s in queries]
queries = [''.join(c for c in s if c not in string.digits) for s in queries]
queries = [value for value in queries if value not in stopwords]
queries = [s for s in queries if s]

abstract = [''.join(c for c in s if c not in string.punctuation) for s in abstract]
abstract = [''.join(c for c in s if c not in string.digits) for s in abstract]
abstract = [value for value in abstract if value not in stopwords]
abstract = [s for s in abstract if s]

# lemmatizes the words so they are similar


#   The IDF dicts will hold all of the words in the array with their values since it will not be unique for each
# iteration of a word

# The TFIDF will initially hold just the TF which is the number of times it appears in that document


words_in_doc = 0
query_document_number = 0
query_TFIDF = {}
#   goes through the document and finds the number of times that a word appears in a query
for i in queries:
    if i == "INDEX":
        #makes sure it's not the very first document
        # if query_document_number != 0:
        #     for j in query_TFIDF[query_document_number].keys():
        #         query_TFIDF[query_document_number][j] = (query_TFIDF[query_document_number][j] / words_in_doc)
        #     words_in_doc = 0
        query_document_number += 1
        # initializes the document to hashtable
        query_TFIDF[query_document_number] = {}
    elif i == "W":
        continue
    elif i in query_TFIDF[query_document_number]:
        query_TFIDF[query_document_number][i] += 1
        words_in_doc += 1
    else:
        query_TFIDF[query_document_number][i] = 1
        words_in_doc += 1

# goes through and divides each TF by the length of the document



query_IDF = {}
# goes through all of the queries if it's in the hashtable then it skips. If not it goes through
# query_hash1 and finds which documents it was in
for i in queries:
    # this means the words document count has not been counted
    if i not in query_IDF:
        # this is going through the documents
        for j in query_TFIDF.keys():
            # if the word is in the document then it's added
            if i in query_TFIDF[j]:
                if i not in query_IDF:
                    query_IDF[i] = 1
                # else we add to the total
                else:
                    query_IDF[i] += 1

# normalizing the IDF so each word has an IDF score
for i in query_IDF.keys():
    query_IDF[i] = numpy.log(query_document_number / query_IDF[i])
# multiplies each word in TFIDF in each document by it's IDF value
for j in query_TFIDF.keys():
    for i in query_TFIDF[j].keys():
        query_TFIDF[j][i] = numpy.log(query_TFIDF[j][i]) * query_IDF[i]

# this is the same version but for abstract
abstract_words_in_doc = 0
abstract_document_number = 0
abstract_TFIDF = {}
# we need to make it so we only grab the words in the abstract
grab = False
for i in abstract:
    # if it's Index this means that we have reached a new document
    if i == "INDEX":
        # if abstract_document_number != 0:
        #     for j in abstract_TFIDF[abstract_document_number].keys():
        #         abstract_TFIDF[abstract_document_number][j] = (abstract_TFIDF[abstract_document_number][j] / abstract_words_in_doc)
        #     abstract_words_in_doc = 0
        abstract_document_number += 1
        abstract_TFIDF[abstract_document_number] = {}
    # this means that we have reached the "W" and after this is the abstract
    elif i == "W":
        grab = True
        continue
    elif i == "T":
        # this means that we have reached the end of the abstract and we cannot grab anything after this
        grab = False
    elif grab:
        if i in abstract_TFIDF[abstract_document_number]:
            abstract_TFIDF[abstract_document_number][i] += 1
            abstract_words_in_doc += 1
        else:
            abstract_TFIDF[abstract_document_number][i] = 1
            abstract_words_in_doc += 1
    else:
        continue

abstract_IDF = {}
for i in abstract:
    # this means the words document count has not been counted
    if i not in abstract_IDF:
        # this is going through the documents
        for j in abstract_TFIDF.keys():
            # if the word is in the document then it's added
            if i in abstract_TFIDF[j]:
                if i not in abstract_IDF:
                    #   this means it has not been added to the list
                    #   so it is initialized
                    abstract_IDF[i] = 1
                else:
                    #   this means that it
                    abstract_IDF[i] += 1

#   normalization
for i in abstract_IDF.keys():
    abstract_IDF[i] = numpy.log(abstract_document_number / abstract_IDF[i])
#   multiplication by IDF scores
for j in abstract_TFIDF.keys():
    for i in abstract_TFIDF[j].keys():
        abstract_TFIDF[j][i] = numpy.log(abstract_TFIDF[j][i]) * abstract_IDF[i]

# output list printing the
outputlist = []

#   this function will return the cosine similarity between two documents
def cosinesimilarity(querydoc, abstractdoc):
    top_array = []
    bottom_array1 = []
    bottom_array2 = []
    array1 = []
    array2 = []

    for a in querydoc.keys():
        if a in abstractdoc:
            array1.append(querydoc[a])
            array2.append(abstractdoc[a])
            # multiplies the values and adds them to list
            top_array.append(querydoc[a] * abstractdoc[a])
            # for the bottom two array squares all values
            bottom_array1.append(math.pow(querydoc[a], 2))
            bottom_array2.append(math.pow(abstractdoc[a], 2))
    # creates the cosine score by summing the top array and multiplying
    # the sums of the bottom two arrays and squaring all
    # this means that the two arrays had no common words
    if sum(bottom_array1) == 0 or sum(bottom_array2) == 0:
        return 0
    # cosine = sum(top_array)/sum(bottom_array1) + sum(bottom_array2)
    cosine = 1 - spatial.distance.cosine(array1, array2)
    return cosine


#   this goes through the list which needs to be printed
#   this goes through the query TFIDF scores by document number
#   we need to do this because all of the indices begin at 1 and not 1
for i in range(len(query_TFIDF) + 1)[1:]:
    #   sets sort list to empty
    sort_list = []
    # goes through the abstract and adds the cosine similarity between this query and every single abstract
    for j in range(len(abstract_TFIDF) + 1)[1:]:
        # i is the query number, j is the abstract number, and then finally there is the cosine similarity
        index = [i, j, cosinesimilarity(query_TFIDF[i], abstract_TFIDF[j])]
        # gets rid of all cosine similarities of 0
        if index[2] == 0:
            continue
        else:
            sort_list.append(index)
    # sorts sort_list based on the second element
    sort_list.sort(key=lambda x: x[2])
    #   adds to output list where i is the number of query
    outputlist.append(sort_list)


with open("output.txt", 'w') as f:
    for i in range(len(outputlist)):
        for j in range(len(outputlist[i])):
            f.write(str(outputlist[i][j][0]) + " " + str(outputlist[i][j][1]) + " " + str(outputlist[i][j][2]))
            f.write("\n")
