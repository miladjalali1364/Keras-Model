
# **Develop your first Deep Learning Model in Python with Keras**
Keras is an open-source API used for solving a variety of modern machine learning and deep learning problems.<br>
  It enables the user to focus more on the logical aspect of deep learning rather than the brute coding aspects.<br>
  Keras is an extremely powerful API providing remarkable scalability, flexibility, and cognitive ease by reducing the user’s workload.<br>
  It is written in Python and uses TensorFlow or Theano as its backend.
  
# **This Keras tutorial has a few requirements:**

    You have Python 2 or 3 installed and configured.
    You have SciPy (including NumPy) installed and configured.
    You have Keras and a backend (Theano or TensorFlow) installed and configured.


# **Table of Contents**

    Overview of Neural Network
    Introduction to Keras
    Step by Step Implementation of your First Keras Model
    Combining all the code
    EndNote

# **Brief Overview of Neural Network**

  Neural Network consists of a larger set of neurons, which are termed units arranged in layers. 
  In simple words, Neural Network is designed to perform a more complex task where Machine Learning algorithms do not find their use and fail to achieve the required performance.

Neural Networks are used to perform many complex tasks including Image Classification, Object Detection, Face Identification, Text Summarization, speech recognition, and the list is endless.

# Build your first Neural Network model using Keras
   We will build a simple Artificial Neural network using Keras step by step that will help you to create your own model in the future.<br>
   
**Step-1) Load Data**<br>

    import pandas as pd
    df = pd.read_csv(r"C:\Users\...\PycharmProjects\data.csv", encoding='utf-8',
                   index_col = False)

**Step-2) Define Keras Model**<br>

Model in Keras always defines as a sequence of layers. It means that we initialize the sequence model and add the layers one after the other which is executed as the sequence of the list. 
Practically we have to try experimenting with the process of adding and removing the layers until we are happy with our architecture.

In this example, We will define a fully connected network with three layers. 
To define the fully connected layer use the Dense class of Keras.
    
Keras offers an Embedding layer that can be used for neural networks on text data.

It requires that the input data be integer encoded, so that each word is represented by a unique integer. This data preparation step can be performed using the Tokenizer API also provided with Keras.

The Embedding layer is defined as the first hidden layer of a network. It must specify 3 arguments:

**It must specify 3 arguments:**

   **input_dim:**
    This is the size of the vocabulary in the text data. For example, if your data is integer encoded to values between 0-10, then the size of the vocabulary would be 11 words.
    
   **output_dim:**
    This is the size of the vector space in which words will be embedded. It defines the size of the output vectors from this layer for each word.
    Forexample, it could be 32 or 100 or even larger. Test different values for your problem.
    
   **input_length:**
    This is the length of input sequences, as you would define for any input layer of a Keras model.For example, if all of your input documents are               comprised of 1000 words, this would be 1000.
    
       tokenize = Tokenizer()
       tokenize.fit_on_texts(Data_Train['clean'])
       #b=tokenize.fit_on_texts(Data_Test['clean'])

       encoded_xtrain = tokenize.texts_to_sequences(Data_Train['clean'])
       encoded_test = tokenize.texts_to_sequences(Data_Test['clean'])
       vocab_size = len(tokenize.word_index) + 1


       encoded_xtrain1 = (pd.get_dummies(Data_Train['Suggestion'])).values.tolist()
       encoded_test1 = (pd.get_dummies(Data_Test['Suggestion'])).values.tolist()

       arr = np.array(encoded_xtrain1)
       arr1 = np.array(encoded_test1)

       Data_Train_p = pad_sequences(encoded_xtrain, maxlen=100, padding='post')
       Data_Test_p = pad_sequences(encoded_test, maxlen=100, padding='post')

       #Create the model Keras
       model = Sequential()
       model.add(Embedding(vocab_size, 100, input_length=100))
       model.add(Conv1D(filters=120, kernel_size=1, activation='relu'))
       model.add(MaxPooling1D(pool_size=4))
       model.add(Flatten())
       model.add(Dense(20, activation='relu')) 
       model.add(Dense(3, activation='softmax')) 
       print(model.summary())

**Step-3) Compile The Keras Model**<br>

When we compile the Keras model, it uses the backend numerical libraries such as TensorFlow or Theano. Whatever backend you are using automatically chooses the best way to represent the network on your hardware such as CPU, GPU, or TPU.

When we are compiling the model we must specify some additional parameters to better evaluate the model and to find the best set of weights to map inputs to outputs.

   **Loss Function** – one must specify the loss function to evaluate the set of weights on which model will be mapped. we will use cross-entropy as a loss function which is actually known as binary cross-entropy used for binary classification.
   
   **Optimizer** – second is the optimizer to optimize the loss. we will use adam which is a popular version of gradient descent and gives the best result in most problems.
   
     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

**Step-4) Start Training (Fit the Model)**

After successful compilation of the model, we are ready to fit data to the model and start training the neural network. Along with providing data to model, we need to define a number of epochs and batch size over which training occurs.

   **Epoch** – one single pass through all the rows in the training dataset
  
   **Batch size** – number of samples considered by the model before updating the weights.
   
      model.fit(Data_Train_p, arr, epochs=50, verbose=2)
One epoch can be comprised of more than one batch. These parameters are finally decided after the heat and trial method.

**Step-5) Evaluate the Model**

    loss, acc = model.evaluate(Data_Test_p, arr1, verbose=0)
    print('Test Accuracy: %f' % (acc * 100))
    print('Test loss: %f' % loss)
To evaluate the model use the evaluate method and pass the input and output to the model and check the performance.


# EndNote

A neural network builds a network of connected layers with multiple neurons in each layer. As we increase the number of layers the network is capable to learn more complex features.

You have easily build your first Neural Network model using Keras. I hope it was easy to catch all the things,

     

