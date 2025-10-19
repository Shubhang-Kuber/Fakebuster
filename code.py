#Step-1
#Import the Tensorflow Library into the Google collab Environment
!pip install tensorflow --upgrade


#Step-2
#Importing all the necessary Libraries into the ipynb
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


#Step-3
#Collab has a built in utility called as "files". We are using this to to upload the downloaded files from the laptop
#Click on the choose file button
#File Explorer will open and then you can upload the file
#Make the file uploaded has the same name as that of the one getting pre-processed in the next code cell
from google.colab import files
uploaded = files.upload()


#Step-4
#From the Pandas library we are the read_csv function. It is used to read the Comma Seperated files
#The entire dataframe collected is stored in the variable called as "data"
#data_head(), as the name suggestes will display the first 5 rows (Known as the head of the CSV file)
#Make sure the the file being read is same as the one that is uploaded

data = pd.read_csv("news (1) (1).csv")
data.head()

#Step-5
#Since the CSV File has one unnamed column, we are going to drop that column
#This Step is used for data Cleaning
#Data Cleaning is the process of removing the rundandancies, NaN values, None Values etc from the files
data = data.drop(["Unnamed: 0"], axis=1)
data.head(5)

#Step-6
#From the pre-processing class we are creating an object of the LabelEncoder
#Converts the text/strings to numerical values
#The object Created is going to learn about all the types of labels like "Positive", "Negative", "Neutral"
#Subsequently The Tags of the label will get a unique number for their representation
#transform function will replace all the tags of the label column to the respective numbers assigned
le = preprocessing.LabelEncoder()
le.fit(data['label'])
data['label'] = le.transform(data['label'])

#Step-7
#Setting up the hyperparameters for the text pre-processing which will later be used for the traing the Neural Network
#In the vector space we will be defining atmost 50 demensions for getting the relationshipd between words
#With each input sequence ; the maximum token that can be forme is of length 50
#If the sequence/sentence has lesser words, then zeroes will be added at the end
# If the sequence/sentence has excess words, then sequence will be cutoff at the end
#Out of Vocabulory holds the words that are unrelated for the model training
embedding_dim = 50
max_length = 54
padding_type = 'post'
trunc_type = 'post'
oov_tok = "<OOV>"
training_size = 3000
test_portion = 0.1


#Step-8
# Extract the first 'training_size' number of titles, texts, and labels from the dataset
# Create and train a tokenizer on the 'title' column to build a vocabulary of unique words
# Get the dictionary (word_index) mapping each word to a numeric ID and find total vocab size
# Convert each title into a sequence of integers using that tokenizer
# Pad all sequences to the same length so the model can process them uniformly

title = []
text = []
labels = []
for x in range(training_size):
    title.append(data['title'][x])
    text.append(data['text'][x])
    labels.append(data['label'][x])

tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(title)
word_index1 = tokenizer1.word_index
vocab_size1 = len(word_index1)
sequences1 = tokenizer1.texts_to_sequences(title)
padded1 = pad_sequences(sequences1, padding=padding_type, truncating=trunc_type)


#Step-9
# Calculate how many samples should go into the test set based on the test_portion ratio
# Split the padded title sequences into training and testing parts
# The first 'split' samples are reserved for testing
# The remaining samples are used for training the model
# Do the same split for the corresponding labels so data and labels stay aligned
split = int(test_portion * training_size)
training_sequences1 = padded1[split:training_size]
test_sequences1 = padded1[0:split]
test_labels = labels[0:split]
training_labels = labels[split:training_size]


#Step-10
#The training and testing sequences are converted into the Numpy arrays 
#Allows the Deep learning models to take in numerical input as Numpy arrays
training_sequences1 = np.array(training_sequences1)
test_sequences1 = np.array(test_sequences1)


#Step-11
#Using the'wget' command we download the GloVe (Global Vectors for Word Representation) zip file from Stanford's NLP dataset repository
#Wget function is used the retrive the file from the URl Mentioned & stores it in the current directory
#Unzip command is used for extracting the contents of the Zip file
!wget https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
!unzip glove.6B.zip


#Step-12
embedding_index = {}
with open('glove.6B.50d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
        
embedding_matrix = np.zeros((vocab_size1 + 1, embedding_dim))

for word, i in word_index1.items():
    if i < vocab_size1:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


#Step-13
# Build a sequential neural network model for text classification
model = tf.keras.Sequential([
    # Embedding layer: maps words to their GloVe vectors; not trainable to keep pre-trained embeddings fixed
    tf.keras.layers.Embedding(vocab_size1 + 1, embedding_dim, input_length=max_length, 
                              weights=[embedding_matrix], trainable=False),
    # Dropout layer to prevent overfitting by randomly ignoring 20% of neurons during training
    tf.keras.layers.Dropout(0.2),
    # 1D Convolution layer: extracts local patterns/features from sequences of word embeddings
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    # MaxPooling layer: reduces dimensionality by taking the maximum value in each pool, summarizing features
    tf.keras.layers.MaxPooling1D(pool_size=4),
    # LSTM layer: captures long-term dependencies and sequence context in the text
    tf.keras.layers.LSTM(64),
    # Dense output layer with sigmoid activation for binary classification (e.g., positive/negative)
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with binary cross-entropy loss, Adam optimizer, and accuracy as the evaluation metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the summary of the model architecture including layers, output shapes, and number of parameters
model.summary()



#Step-14
history = model.fit(
    training_sequences1, 
    np.array(training_labels), 
    epochs=50, 
    validation_data=(test_sequences1, np.array(test_labels)), 
    verbose=2
)



#Step-15
X = "Karry to go to France in gesture of sympathy"

sequences = tokenizer1.texts_to_sequences([X])
sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
if model.predict(sequences, verbose=0)[0][0] >= 0.5:
    print("This news is True")
else:
    print("This news is False")


#Step-16
# Example 1
X = "The government announced new tax reforms today"
sequences = tokenizer1.texts_to_sequences([X])
sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
if model.predict(sequences, verbose=0)[0][0] >= 0.5:
    print("This news is True")
else:
    print("This news is False")



# Example 2
X = "Aliens landed in central park last night according to eyewitnesses"
sequences = tokenizer1.texts_to_sequences([X])
sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
if model.predict(sequences, verbose=0)[0][0] >= 0.5:
    print("This news is True")
else:
    print("This news is False")



#Example 3
X = "Local school won the national robotics competition"
sequences = tokenizer1.texts_to_sequences([X])
sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
if model.predict(sequences, verbose=0)[0][0] >= 0.5:
    print("This news is True")
else:
    print("This news is False")
