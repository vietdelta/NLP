import numpy 
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np 
import random
from random import choice
from sklearn.preprocessing import normalize
from utils import *

class LDA():
    def __init__(self,alpha = 0.5,beta = 0.5,num_topic = 10,max_iter = 20):
        self.alpha = alpha
        self.beta = beta
        self.num_topic = num_topic
        self.max_iter = max_iter
    def preprocess(self,file_path):
        '''
        Input: A file path - Numpy array
        Output: The value parameters
        '''
        corpus = []
        # Read the text file
        with open(file_path, 'r') as f:
            line = f.readline()
            while(line):
                if(line[0]=="*"):
                    pass
                else:
                    corpus.append(remove_stop_words(line))
                line = f.readline()
        vectorizer = CountVectorizer()
        # Turn the raw corpus into a BOW matrix
        self.BOW_matrix = vectorizer.fit_transform(corpus)
        # Get the vocabulary
        self.vocab = vectorizer.get_feature_names()
        # Initiallize the parameters for LDA for this corpus
        self.topic_document_table = np.zeros((self.num_topic,self.BOW_matrix.shape[0]))
        self.word_topic_table = np.zeros((self.BOW_matrix.shape[1],self.num_topic))
        self.topic_assignment = []
        self.topic_count = np.zeros((self.num_topic,))
        # Random assign a topic for each word in the corpus 
        for doc_num,x in enumerate(self.BOW_matrix):
            sentence = []
            for i in range(len(x.indices)):
                for k in range(x.data[i]):
                    topic = random.randint(0, self.num_topic-1)
                    self.topic_count[topic]+=1
                    sentence.append([x.indices[i],topic])
                    self.topic_document_table[topic,doc_num]+=1
                    self.word_topic_table[x.indices[i],topic]+=1
            self.topic_assignment.append(sentence)
    def fit(self):
        vocab_length = self.BOW_matrix.shape[1]
        for epo in range(self.max_iter):
            log_likelihood = 0
            print("Epoch ",epo,":")
            for num_doc,doc in enumerate(self.topic_assignment):
                for word_ele in doc:
                    word_ind = word_ele[0]
                    topic_assigned = word_ele[1]
                    # Decrease the parameters
                    self.topic_document_table[topic_assigned,num_doc]-=1
                    self.word_topic_table[word_ind,topic_assigned]-=1
                    self.topic_count[topic_assigned]-=1
                    # Calculate the propability for each word given the others
                    doc_like_topic = (self.topic_document_table[:,num_doc]+self.alpha)/(len(doc)-1+self.alpha*self.num_topic)
                    word_like_topic = (self.word_topic_table[word_ind,:]+self.beta)/(self.topic_count+self.beta*vocab_length)                    
                    p = doc_like_topic*word_like_topic
                    p/=np.sum(p)
                    new_topic = np.random.multinomial(1,p).argmax()
                    log_likelihood+=np.log(p[new_topic])
                    word_ele[1] = new_topic
                    # Increase the parameters
                    self.topic_document_table[new_topic][num_doc]+=1
                    self.word_topic_table[word_ind][new_topic]+=1
                    self.topic_count[new_topic]+=1
                if(num_doc%1000==0 and num_doc>0):
                    print(num_doc)
                    print(log_likelihood/num_doc)
            print(log_likelihood/num_doc)
        self.word_topic_table = normalize(self.word_topic_table,axis=0,norm='l1')
    def get_topic_word(self,no_topic,num_word=10):
        if(no_topic>=len(self.topic_count)):
            print("no_topic is exceed the real number of topics")
            return 0
        sort_arr = np.argsort(self.word_topic_table[:, [no_topic]],axis = 0)
        b = sort_arr[-num_word:]
        for i in range(len(b)):
            print(self.vocab[int(b[i])])




