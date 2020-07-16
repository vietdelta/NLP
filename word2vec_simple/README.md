# WORD2VEC simple implementation using Skip-gram

## Installation
Run 
```pip install -r requirements.txt```
## Usage
You can take a look on the word2vec_example Notebook to understand how to run this implementation.

Create a new word2vec model and initiallize the parameters
```
from word2vec import Word2Vec
w2v = Word2Vec(epochs=10,window_size=3,lr=0.01)
```
You can change the parameters to customize your training

To run this model, you need to pass a python list of sentences (or short documents) into it.
```
file_text = "./Vietnamese_FaceBook_October_processed.txt"
f = open(file_text,"r")
sentences = []
for line in f:
    if(len(line)>1):
        sentences.append(line)
w2v.generate_data(sentences)
```

After that, the word2vec model will help you generate the data points that need for the Skip-gram training process
All you need to do is run it by using the command:

```
w2v.fit()
```

After the training phase, you can find the similar words in the corpus
```
print("Some of the similar words")
print(w2v.most_similar("con_cái"))
print("________________")
print(w2v.most_similar("phụ_nữ"))
print("________________")
print(w2v.most_similar("word"))
print("________________")
print(w2v.most_similar("microsoft"))
print("________________")
print(w2v.most_similar("đông_nhi"))
print("________________")

```