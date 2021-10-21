import numpy, os
import tensorflow as tf
from tensorflow import keras

class neural_network_model:
    def __init__(self, model_name, number_of_metrics):
        self.name         = model_name
        self.metric_count = number_of_metrics
        self.model_path   = "neural_network_models/" + model_name

        if(os.path.isfile(self.model_path)):
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            print("No model found! Please train a model first")
            self.model = None

    # Call this function to train a model
    def train(self, training_data, training_labels, test_data, test_labels, batch_size, epochs):
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape = self.metric_count),
            tf.keras.layers.Dense(self.metric_count * 10, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.metric_count * 3, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation='relu')
        ])

        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
                      loss='mean_squared_error',
                      metrics=['accuracy'])

        model.fit(training_data, training_labels, batch_size, epochs)
        model.evaluate(test_data, test_labels, verbose=2)
        # Save the model to the models folder so it can be re-used later without need for training
        model.save(self.model_path, overwrite=True)

    # Call this function to use an existing model to predict a value
    def predict(self, data):
        # Check if the model has been loaded properly
        if(self.model == None):
            print("Warning: This model has not yet been trained. Please train the model first.")
            return None
        
        # Use the model to predict a value based on the given data. 
        prediction = self.model.predict(data)
        return prediction