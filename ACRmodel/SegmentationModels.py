#!/usr/bin/env python3
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
import tensorflow
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing

class SegmentationCRNN():
    """
    CRNN model to find chord changes. 
    Using datasets in format (n_sequences, n_frames, n_features, 1)
    With chord targets, same as a CRNN ACR models, that will be preprocess inside the model. 
    """
    def __init__(self, input_shape):
        # Create model
        model = tensorflow.keras.models.Sequential()

        # Feature Extractor
        model.add(tensorflow.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape,padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Conv2D(16, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Conv2D(16, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.MaxPooling2D((1,3),padding='same'))
        model.add(tensorflow.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.MaxPooling2D((1,3),padding='same'))
        model.add(tensorflow.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.MaxPooling2D((1,4),padding='same'))
        model.add(tensorflow.keras.layers.Conv2D(80, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Conv2D(80, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        _, a1, a2, a3 = model.output_shape
        
        # Classifier - RNN
        model.add(tensorflow.keras.layers.Reshape((a1, a2*a3), input_shape=(a1, a2, a3)))
        model.add(tensorflow.keras.layers.Bidirectional(
            tensorflow.keras.layers.LSTM(96, return_sequences=True))
        )
        model.add(
            tensorflow.keras.layers.Dense(2, activation='softmax')
        )

        # Compile model
        model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(),
            loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )


        model.summary()
        self.model = model
        print("[INFO] The CRNN model was successfully created.")


    def fit(self, data, targets, dev_data, dev_targets, epochs=50):
        # Get chord changes from chord labels
        changes_targets = SegmentationCRNN.labels2changes(targets)
        dev_changes_targets = SegmentationCRNN.labels2changes(dev_targets)

        # Train model
        self.history = self.model.fit(
            data, changes_targets, epochs=epochs,
            validation_data=(dev_data, dev_changes_targets)
        )
        print("[INFO] The CRNN model was successfully trained.")
    

    @staticmethod
    def labels2changes(targets):
        seg_targets = []

        # Find chord changes
        last_chord = 0
        for sequence in targets:
            new_seg_sequence = []
            for chord in sequence:
                if last_chord == chord:
                    new_seg_sequence.append(0)
                else:
                    new_seg_sequence.append(1)
                    last_chord = chord
            seg_targets.append(np.array(new_seg_sequence))

        return np.array(seg_targets)

    @staticmethod
    def get_change_points(target_changes, sequence_length_ms):
        change_points = []
        _, n_frames, _ = target_changes.shape
        for sequence in target_changes:
            change_points.append(
                np.where(sequence == 1) * (sequence_length_ms / n_frames)
            )

        return change_points

    def score(self, data, targets):
        # Get chord changes from chord labels
        changes_targets = SegmentationCRNN.labels2changes(targets)
        # Evaluate model 
        _, test_acc = self.model.evaluate(data, changes_targets, verbose=2)
        return test_acc
    
    def display_training_progress(self):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.0, 1])
        plt.legend(loc='lower right')
        plt.show()