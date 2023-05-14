# 3M-NLP-Chatbot

* Project Title: Topic-Based Empathic Chatbot
* Description:  The 3M-NLP Chatbot is an AI-based conversational agent capable of engaging in general talk while also offering factual information on Reddit topics encompassing healthcare, education, environment, politics, and technology. It uses advanced natural language processing techniques, notably Named Entity Recognition, Sentiment Analysis, and we fine-tune the pre-trained models to comprehend user queries and generate appropriate responses.
* Getting Started: 

!pip install transformers
!pip install tensorflow_text
!pip install sentence_transformers
!pip install profanity-check
!pip install transformers
!pip install better_profanity
!pip install flask_cors
!pip install flask_ngrok
!pip install pyngrok==4.1.1

Next step is to get the authtoken from ngrok.
Login to ngrok https://ngrok.com/ and get the authtoken

!ngrok authtoken <authtoken>

You are set for hosting the flask app using ngrok.

* Data pre-processing

To create a well structured data for further processing, we firstly take all the raw dataset.
Chitchat dataset: https://github.com/BYU-PCCL/chitchat-dataset
Empathetic dataset: https://github.com/facebookresearch/EmpatheticDialogues
Reddit dataset: We extract the submissions and comments from reddit using the psaw API. The reddit_data_extraction.ipynb has the code for extracting topic based submissions and respective comments.

The BERT_classifier_data_pre.ipynb has the code to preprocess all three datasets into a query-reply format. The Data_combining.ipynb combines the required fields from each dataset and agglomerates it into a single dataset. It has the training of a BERT classifier that determines which type of reply (chitchat vs reddit) is expected from the chatbot. We get the classifier model.
Sentiment_NER_GPT2.ipynb performs some basic data cleaning and contains the code to import the pretrained GPT-2 model, perform NER and Sentiment analysis on the query, refine the query for fine-tuning of the GPT-2 model. The combined dataset is divided into batches of 64, split for getting train and test data and further training the model with custom dataset. We get the fine-tuned GPT-2 model.

* Flask App
3m-NLP_flask_app.ipynb is the main file that integrates the three models and runs the flask app. It loads the fine-tuned GPT-2 model, BERT classifier model and has the RoBERTa pipeline for question answering. Once we run the app we get the interface for our chatbot. The entered query follows the pipeline by refining it and passing it to the loaded GPT-2 model. We get the context using cosine similarity function and append the GPT-2 reply to it. The query and context is passed through the RoBERTa pipeline to get the final reply.

* Evaluation
The code for evaluation methods (BLEU & BERT) can be found in Sentiment_NER_GPT2.ipynb.




