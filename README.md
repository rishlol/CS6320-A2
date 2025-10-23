# starter code for a2

Add the corresponding (one) line under the ``[to fill]`` in ``def forward()`` of the class for ffnn.py and rnn.py

Feel free to modify other part of code, they are just for your reference.

---

One example on running the code:

**FFNN**

``python ffnn.py --hidden_dim 10 --epochs 1 ``
``--train_data ./training.json --val_data ./validation.json``


**RNN**

``python rnn.py --hidden_dim 32 --epochs 10 ``
``--train_data training.json --val_data validation.json``


2.1 FFNN 
Implementation of FFNN.py:forward()

For the FFNN model, our task was to implement the forward pass, which handles the classification of a single, pre-aggregated input vector (like a BoW or averaged embedding). We wired up the layers in the standard two-layer architecture:

Hidden Layer: I took the input vector, ran it through the first linear layer (self.W1), and immediately applied the ReLU activation function (self.activation) to get our hidden representation.

Output Layer: The hidden output was then fed into the final linear layer (self.W2), which maps the hidden dimension to the 5 output classes.

Probabilities: Finally, I applied the LogSoftmax function (self.softmax) to convert the raw scores into log-probabilities, using .view(1, -1) to ensure the shape is correct for the loss function.

Code Snippet:

Python

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        hidden = self.activation(self.W1(input_vector))
        # [to fill] obtain output layer representation
        output = self.W2(hidden)
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output.view(1, -1))
        return predicted_vector
Other Code Understandings (Optimizers, Stopping, etc.):

The model framework uses the Adam optimizer, which is pretty standard because it’s fast and handles gradient updates automatically. For training stability, we’re using nn.NLLLoss() (Negative Log Likelihood Loss) paired with the LogSoftmax output. The cool part is the early stopping mechanism: training stops if the validation accuracy on the dev set starts dropping while training accuracy is still going up. This is essential for preventing the model from overfitting to the training data.

2.2 RNN (25pt)
Implementation of RNN.py:forward()

The RNN implementation was trickier because we had to handle sequences, not just single vectors. The goal of the forward function here is to process a sequence of word vectors (one word at a time) and use the final output to classify the whole review.

Recurrence: I used PyTorch's built-in self.rnn(inputs). This function is key because it processes the whole sequence and outputs two things: the full output sequence (which we ignore using _) and the final hidden state (hidden), which summarizes the entire input.

Classification Prep: I grabbed the final hidden state (hidden[-1]), which is the review's "summary," and flattened it into a shape suitable for the linear classifier (final\_hidden = hidden[-1].view(1, -1)).

Classification: This final hidden vector is passed through our single linear classification layer (self.W) and then converted to log-probabilities using self.softmax.

Code Snippet:

Python

    def forward(self, inputs):
        # [to fill] obtain hidden layer representation
        _, hidden = self.rnn(inputs)
        # [to fill] obtain output layer representations
        final_hidden = hidden[-1].view(1, -1)
        # [to fill] sum over output 
        output = self.W(final_hidden)
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output)
        return predicted_vector
Key Differences vs. FFNN:

The main thing functioning differently is how the input is handled.

FFNN takes an already summarized vector of the whole review. It has no idea of word order.

RNN takes the input as a sequence ((seq\_len, 1, input\_dim)), processes it word-by-word, and uses its hidden state to maintain a running "memory" of the review. This lets it capture context and word order, which the FFNN completely misses. We only use the final hidden state (the one after the last word) for the actual prediction.
