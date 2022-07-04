```python

from learntools.deep_learning_intro.dltools import animate_sgd

# YOUR CODE HERE: Experiment with different values for the learning rate, batch size, and number of examples
learning_rate = 0.08
batch_size = 256
num_examples = 256

animate_sgd(
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_examples=num_examples,
    # You can also change these, if you like
    steps=50, # total training steps (batches seen)
    true_w=3.0, # the slope of the data
    true_b=2.0, # the bias of the data
)

```

https://www.kaggle.com/neuralneuron/exercise-stochastic-gradient-descent/edit