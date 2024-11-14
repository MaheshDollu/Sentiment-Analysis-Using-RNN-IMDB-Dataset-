# Sentiment-Analysis-Using-RNN-IMDB-Dataset-

The code implements a deep learning model for sentiment analysis using the IMDB dataset. The dataset contains movie reviews labeled as positive or negative, and the goal is to predict the sentiment of a given review. Here's an explanation of each step in the implementation:

# 1. Import Libraries

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.datasets import imdb

The code imports TensorFlow and relevant Keras modules for building and training the neural network, including layers like Embedding, LSTM, and Dense, as well as functions for preprocessing the dataset.

# 2. Load and Preprocess the IMDB Dataset


vocab_size = 10000  # Use top 10,000 words

max_len = 200  # Max length of reviews

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

x_train = pad_sequences(x_train, maxlen=max_len)

x_test = pad_sequences(x_test, maxlen=max_len)

IMDB dataset: This dataset is a collection of 50,000 movie reviews labeled as positive (1) or negative (0).
num_words=vocab_size: This limits the dataset to the top 10,000 most frequent words in the reviews. This ensures that the model focuses on the most relevant vocabulary.
Padding: The reviews in the dataset are of varying lengths, so pad_sequences() is used to ensure that all reviews have the same length (max_len=200). If a review is shorter than 200 words, it is padded with zeros; if it is longer, it is truncated.

# Build the RNN Model with LSTM
python
Copy code
model = Sequential([
   
    Embedding(vocab_size, 128, input_length=max_len),
    
    LSTM(64, dropout=0.5, recurrent_dropout=0.5),
    
    Dense(1, activation='sigmoid')  # Binary classification (positive/negative)
])

Sequential(): A simple linear stack of layers.
Embedding(vocab_size, 128, input_length=max_len): This layer transforms the input word indices into dense vectors of fixed size (128). It takes input sequences of integers (representing words) of length max_len (200) and maps them to 128-dimensional vectors. The layer learns the word embeddings during training.

LSTM(64, dropout=0.5, recurrent_dropout=0.5): This is a Long Short-Term Memory (LSTM) layer, which is a type of RNN used for sequence processing. It has 64 units (memory cells), with a dropout rate of 50% (dropout=0.5) to prevent overfitting and recurrent dropout (recurrent_dropout=0.5) to regularize the LSTM connections.

Dense(1, activation='sigmoid'): A fully connected output layer with a single neuron. The activation function sigmoid is used for binary classification (positive or negative sentiment). The output will be a value between 0 and 1, which can be interpreted as the probability of the review being positive.
# Compile the Model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

optimizer='adam': Adam is an efficient optimization algorithm that adapts the learning rate during training. It is commonly used for training deep learning models.

loss='binary_crossentropy': The loss function used for binary classification tasks. It measures the difference between the predicted probability and the actual label (0 or 1).

metrics=['accuracy']: Accuracy is tracked as the metric to evaluate the model’s performance.

# Train the Model

model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

epochs=5: The model will be trained for 5 complete iterations through the training data.

batch_size=64: The model will update its weights after every 64 samples (mini-batch).

validation_data=(x_test, y_test): This argument provides a validation set on which the model will be evaluated after each epoch. It helps in monitoring the model’s performance on unseen data and detecting overfitting.

# Evaluate the Model

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print(f"Test accuracy: {test_acc}")

model.evaluate(): This function evaluates the trained model on the test data (x_test, y_test), returning the loss and accuracy. The verbose=2 setting prints a minimal output during evaluation.

Test accuracy: The final accuracy of the model on the test set is printed.


# Summary:

This implementation uses an LSTM-based Recurrent Neural Network (RNN) to classify sentiment in movie reviews from the IMDB dataset.
It preprocesses the reviews by limiting vocabulary size and padding the sequences to a fixed length.
The model consists of an embedding layer, an LSTM layer with dropout regularization, and a dense output layer with a sigmoid activation function.
The model is compiled using the Adam optimizer and binary cross-entropy loss and is trained for 5 epochs, evaluating performance on the test set.
