# TF-IDF Feature Extraction
## **Data Preprocessing**

1.  Remove uninformative rows
	- drop rows with null Subject or Body
2.  Text cleaning
3.  Remove uninformative sentences, 
	- like “Notice: This message was sent from outside the University of Victoria email system. Please be cautious with links and sensitive information.”
4.  Remove /r, /n
5.  Handle email and url addresses, since it’s not strongly related to english words
6.  Text tokenization
7.  Remove stopwords (NLTK)
8.  Lemmatization (root words)

## **Feature Extraction**

1.  Combine ‘Subject’ and ‘Body’ into a single text for an email
2.  Applied TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text into numerical features.

|  access    | account        | action         | address       |attachment       | automatically|
|----------- |---------------|-------------- |-------------- |------------------|-----------|
|0 |0 |0.225402147 |0.327260556 |0 |0            |
|0 |0 |0 |0 |0|0|
|0.070280488 |0 |0 |0 |0 |0 |
|0 |0.430047625 |0 |0 |0 |0 |
...
