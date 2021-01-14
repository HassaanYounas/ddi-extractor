# DDI-Extraction
___
## How to Run
Running the app is very simple. This is a Flask app, so you need Python for this. Just create a virtual environment for Python, install the dependencies from the requirements.txt file and start the Flask app.
___
Some information about the directories:
### ai-models
This directory contain all the machine learning models. We are using three different models:
* Named Entity Recognition Model (Stanford CoreNLP)
* Word2Vec Model (Gensim)
* Bi-LSTM Neural Network (Keras)
### ddi/ner
This directory contains the Python script to generate the training data from the corpus. It also contains the corenlp.jar file required for training the NER model. After the Python script has been run, we can run the command from 'command.txt' file to train the model.
### ddi/re
This contains the scripts to prepare the training and testing data for the neural network. First, we run the 'data.py' script to generate the test data. Then we run the 'network.py' for training the network. We will use the 'testing.py' script after training the network.
### java
This contains the Stanford CoreNLP libraries and command to train the NER model.
### static
This contains the CSS and JS for the frontend.
### templates
This contains the HTML files for the frontend.
## NOTE-1
There is supposed to be a directory inside 'ddi' called 'corpora'. This should contain the corpus in two segments, 'training', 'testing'. The names of the directories should be as given. Since, the training corpus is very large, I could not push it to the repository. We went for an 80-20 split for the training and testing data.

## NOTE-2
BeautifulSoup, TensorFlow, CSV, NLTK, Keras, NumPy and Gensim need to installed for Python.
