```python

data\_dir = 'data' batch\_size = 1000000 img\_height = 224 img\_width =
224

ds\_train = tf.keras.preprocessing.image\_dataset\_from\_directory(
data\_dir, validation\_split=0.2, subset="training", seed=123,
image\_size=(img\_height, img\_width), batch\_size=batch\_size)

ds\_valid = tf.keras.preprocessing.image\_dataset\_from\_directory(
data\_dir, validation\_split=0.2, subset="validation", seed=123,
image\_size=(img\_height, img\_width), batch\_size=batch\_size)


```