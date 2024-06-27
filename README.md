# Summer-2024-ECE-597-Group8
# Individual Contribution

## Yue Cao
- Week 1: Host Discussion meeting, study 597 guidelines and proposed questions
- Week 2: literature review, search and read papers related to phishing email and data characteristics, and summarize into a doc; discuss with teammates and exchange ideas
- Week 3: data preprocessing and extract features by using TF-IDF
- Week 4:
  - extract features by using bag-of-words,
  - encapsulate TF-IDF feature extraction codes
  - Bag-of-word feature extract codes implement and encapsulate
- Week 5:
  - add feature extraction codes for email length
  - Train model with balanced data ratio in the following scenarios:
  - Random forest with TF-IDF feature dataset (with LDR, without LDR)
  - Random forest with BoW feature dataset (with LDR, without LDR)
- Week 6:
  - add word2Vec feature extraction codes
  - initial model training Random forest, naive bayes. train with 3 types of features: TFIDF, BoW, word2Vec. and apply techniques like: pca and fld to transform features to see if can get better or worse results.

## Zixia Li
- Week 1: Attend seminars, study 597 guidelines and ask questions
- Week 2: Find literature, search and read papers related to phishing emails and data characteristics, summarize ideas and sort out ideas, discuss and exchange ideas with teammates
- Week 3: Preprocess data and extract useful features such as JS, html, link, address, keywords.
- Week 4: After group discussion, the address and link were deleted. They were already processed data, and the feature was finally extracted into JS and HTML for encapsulation.
- Week 5: Change the previous code. On the basis that the two features of html and JS only display F and T, if there are several, it will become a numeric type
  - Train the model with balanced data proportions in the following scenarios:
  - Decision tree with TF-IDF feature dataset
  - Decision tree with BoW feature dataset.
- Week 6:Initial model training decision tree, CNN. Trained using 3 types of features: TFIDF, BoW, word2Vec. And apply techniques like pca and fld to transform the features and see if you can get better or worse results.

## Xinyi Chen
- Week 1- Attend seminars, study 597 guidelines and ask questions
- Week 2- Find literature, search and read papers related to phishing emails and data characteristics, summarize ideas and sort out ideas, discuss and exchange ideas with teammates
- Week 3-Preprocess data and extract useful features for languages ​​like , French, and others.         
- Week 4 - Modify features, extract JS, HTML, and encapsulate them. Make the original data set which has normal emails 10 times larger than phishing emails, and label them.
- Week 5 - Change the previous code. On the basis that the two features of html and JS only display F and T, if there are several, it will become a numeric type. Build SVM model with k-fold validation.
- Week 6 - Train the SVM and the RNN models by using three types of features respectively(Tfidf, Bow, word2Vector). Compare the performance of models and feature types through Accuracy, ROC-AUC Score, Pricision, Recall, Average Precision-Recall Score, F1 Score and Balanced Accuracy.

## Feiyi Xie
- Week 1: Attend seminars, study 597 guidelines and ask questions
- Week 2: Literature review
- Week 3: Feature Extraction: Extract Homoglyphs features. 
- Week 4: Combine the feature extraction functions and generate processed datasets. 
- Week 5:
  - Do data analysis and visualization on all features except BOW and TF-IDF. 
  - Implement the training pipeline template (based on Guo’s work), experiment a few transformers on features and TF-IDF features (using SVM). 
  - Propose to change boolean type features to number count, adopt the code to this change. 
## Weilong Qian

- Week 1: Attend seminars and ask questions
- Week 2: Literature review&summerize
- Week 3: Data preprocessing and extracting Keyword feature by using TF-IDF
- Week 4: Testing the performance of Keyword feature by changing the feature number
- Week 5:  Create instruments file for members not familiar with models  

## Mingxi Guo

- Week 1: Attend seminars, study 597 guidelines and ask questions
- Week 2: Literature review
- Week 3: Feature Extraction: abnormal number
- Week 4: modify the feature to adjust performance
- Week 5:  implement the training template


