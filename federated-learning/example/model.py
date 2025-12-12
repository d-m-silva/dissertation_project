import tensorflow as tf


def model_adult(learning_rate):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(10,)))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return model


def model_sent140(learning_rate):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape=(200,), activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(86, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return model
