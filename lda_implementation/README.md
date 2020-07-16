
### 
# Requirements
Run ```pip install -r requirements.txt```
# Usage
You can take a look of the Example Notebook.

Initialization:

```
from lda2py import LDA
lda = LDA_denvi(num_topic=20,max_iter=50)
```

Preprocessing text file:
```
file_path = "./VietNameseFacebookNovember.txt"
lda.preprocess(file_path)
```

Run the LDA model:
```
lda.fit()
```

You can see what are the most popular words for each topic:
```
lda.get_topic_word(no_topic=8,num_word=15)
```

This implementation is for Vietnamese dataset, you can customize the preprocessing stage for your language by changing details in utils.py

# Contact
You can contact me at hoangvietit15@gmail.com for any details or questions related to this project





