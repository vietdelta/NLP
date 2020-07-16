from collections import defaultdict
import numpy as np
from sklearn.utils import shuffle
from scipy.spatial.distance import cosine

class Word2Vec():
    def __init__(self, window_size=2,dim=20, epochs=10, lr=0.01):
        self.window_size = window_size
        self.dim = dim
        self.epochs = epochs
        self.lr = lr
        # self.vocab_size = self.get_vocab_size
    def get_vocab_size(self,vocab_matrix):
        return vocab_matrix.shape[1]
    def fit(self):
        #Initialize the W1 and W2 matrix
        self.w1 = np.random.rand(self.dim,self.vocab_size)*0.01
        self.w2 = np.random.rand(self.vocab_size, self.dim)*0.01
        print("Training....")
        for e in range(self.epochs):
            loss = 0
            train_data = shuffle(self.training_data)
            for i,data_point in enumerate(train_data):
                word_ind = data_point[0]
                word_oh = self.get_onehot(word_ind)

                #Foward pass:
                Z1 = self.w1.dot(word_oh)
                A1 = np.maximum(Z1,0)
                Z2 = self.w2.dot(A1)
                Y_hat = self.softmax(Z2)

                contexts = data_point[1:]
                for context in contexts:
                    
                    #Back propagation:
                    context_oh = self.get_onehot(context)
                    E2 = Y_hat-context_oh
                    dW2 = E2.dot(A1.T)
                    E1 = np.dot(self.w2.T,E2)
                    E1[Z1 <= 0] = 0
                    dW1 = E1.dot(word_oh.T)

                    #Update:
                    self.w1 += -self.lr*dW1
                    self.w2 += -self.lr*dW2
                    # print(Y_hat)
                    # print(context)
                    loss += self.cross_entropy(Y_hat,context_oh)
                if(i%200==0):
                    print("Data_point number ",i," loss = ",loss/i) 
            print("Epoch "+str(e)+" Loss = "+str(loss/len(self.training_data)))               
        return 0
    def cross_entropy(self,prediction, target, epsilon=1e-12):
        '''
        Input: Prediction and the target (or ground_truth) vectors - Numpy array
        Output: Cross entropy value - Scalar
        '''
        prediction = np.clip(prediction, epsilon, 1. - epsilon)
        # print(prediction)
        N = prediction.shape[0]
        ce = -np.sum(target*np.log(prediction+1e-9))/N
        return ce
    def softmax(self,x):
        '''
        Input: A vector - Numpy array
        Output: The value of vector after applied softmax - Numpy array
        '''
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    def generate_data(self,sentences):
        '''
        Before training word2vec model, we have to prepare a set of training data. A training point in the set have have the indices 
        of the anchor word and the context words in following order:
        training point = (anchor_word, context_word_0, context_word_1, context_word_2,...,context_word_n)

        Input: A list of sentences - Python list
        Output: Training data for word2vec. - Python tuples

        '''
        self.training_data = []
        word_counts = defaultdict(int)
        for sent in sentences:
            for word in sent.split():
                word_counts[word]+=1
        self.vocab_size = len(word_counts.keys())
        self.vocab = list(word_counts.keys())
        self.word_index = dict((word,i) for i,word in enumerate(self.vocab))
        self.index_word = dict((i,word) for i,word in enumerate(self.vocab))
        for num,sent in enumerate(sentences):
            print("Processing "+str(num)+"/"+str(len(sentences))+"......")
            sents = sent.split()
            for i,word in enumerate(sents):
                word_ind = self.vocab.index(word)
                anchor = word_ind
                training_point = []
                training_point.append(anchor)
                for ind in range(i-self.window_size,i+self.window_size+1):
                    if(ind>=0 and ind<len(sents) and ind!=i):
                        # print(sents)
                        # print(ind)
                        context_word = sents[ind]
                        context = self.vocab.index(context_word)
                        # context = ind
                        training_point.append(context)
                self.training_data.append(np.array(training_point))
                del training_point
    def get_onehot(self,ind):
        '''
        Input: Index of the word - A number
        Output: A one-hot vector (as a numpy array)
        '''
        vector = np.zeros((self.vocab_size,1))
        vector[ind][0] = 1
        return vector
    def most_similar(self,word):
        '''
        Input: A word - Python string
        Output: The word that is the most similar to the input word (if the input word does exist in the vocabulary)
            or -1 if the input word doesn't exist in the vocabulary
        '''
        try:
            word_ind = self.vocab.index(word)
            print(self.vocab[word_ind])
        except:
            print("This word is not in the corpus")
            return -1
        min_dis = 9999
        ind = 0
        word_oh = self.get_onehot(word_ind)
        word_vector = self.w1.dot(word_oh)
        for i in range(len(self.vocab)):
            if(word == self.vocab[i]):
                continue
            compare_oh = self.get_onehot(i)
            vector = self.w1.dot(compare_oh)
            sim_dis = cosine(word_vector,vector)
            # sim = np.dot(word_vector.T, vector)/(np.linalg.norm(word_vector)*np.linalg.norm(vector))
            # print(sim)
            if(min_dis>sim_dis):
                min_dis = sim_dis
                ind = i
        return self.vocab[ind]