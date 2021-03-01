#!/usr/bin/env python3
import mir_eval
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
import tensorflow
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np

class MLP():
    def __init__(self):
        self.model = MLPClassifier(max_iter = 500)
        print("[INFO] The MLP model was successfully created.")

    def fit(self, data, targets):
        self.model.fit(data, targets)
        print("[INFO] The MLP model was successfully trained.")

    def score(self, data, targets):
        return self.model.score(data, targets)
    
    def display_confusion_matrix(self, data, targets):
        display_labels = np.array(["N", "C", "C:min", "C#", "C#:min", "D", "D:min", "D#", "D#:min", "E", "E:min", "F", "F:min", "F#", "F#:min", "G", "G:min", "G#", "G#:min", "A", "A:min", "A#", "A#:min", "B", "B:min"])
        labels = np.array([i for i in range(len(display_labels))])
        plot_confusion_matrix(estimator=self.model, X=data, y_true=targets,
            labels=labels, display_labels=display_labels,
            normalize='all', xticks_rotation='vertical', include_values=False)  
        plt.show()  



    @staticmethod
    def mir_score(x, y):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(x)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(y)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(), ref_intervals.max(), mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels, est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.triads(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score


class CNN():
    def __init__(self, input_shape, output_classes):
        # Create model
        model = tensorflow.keras.models.Sequential()

        # Feature Extractor
        model.add(tensorflow.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Conv2D(16, (3,3), activation='relu'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Conv2D(16, (3,3), activation='relu'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.MaxPooling2D((1,3)))
        model.add(tensorflow.keras.layers.Conv2D(32, (3,3), activation='relu'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Conv2D(32, (3,3), activation='relu'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Conv2D(32, (3,3), activation='relu'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.MaxPooling2D((1,3)))
        model.add(tensorflow.keras.layers.Conv2D(64, (3,3), activation='relu'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Conv2D(64, (3,3), activation='relu'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.MaxPooling2D((1,4)))
        model.add(tensorflow.keras.layers.Conv2D(80, (3,3), activation='relu'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Conv2D(80, (3,3), activation='relu'))
        model.add(tensorflow.keras.layers.BatchNormalization())

        # Classifier
        model.add(tensorflow.keras.layers.Flatten())
        model.add(tensorflow.keras.layers.Dense(100, activation='relu'))
        model.add(tensorflow.keras.layers.Dense(output_classes, activation='softmax'))

        # Compile model
        model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.1),
            loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )


        model.summary()
        self.model = model
        print("[INFO] The CNN model was successfully created.")

    def fit(self, data, targets, dev_data, dev_targets):
        # Train model
        self.history = self.model.fit(
            data, targets, epochs=10, 
            validation_data=(dev_data, dev_targets)
        )
        print("[INFO] The CNN model was successfully trained.")

    def score(self, data, targets):
        _, test_acc = self.model.evaluate(data, targets, verbose=2)
        return test_acc
    
    def display_training_progress(self):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.0, 1])
        plt.legend(loc='lower right')
        plt.show()


    def display_confusion_matrix(self, data, targets):
        # Define labels
        display_labels = np.array(["N", "C", "C:min", "C#", "C#:min", "D", "D:min", "D#", "D#:min", "E", "E:min", "F", "F:min", "F#", "F#:min", "G", "G:min", "G#", "G#:min", "A", "A:min", "A#", "A#:min", "B", "B:min"])
        labels = np.array([i for i in range(len(display_labels))])

        # Generate predictions
        predictions = tensorflow.argmax(self.model.predict(data), axis=1)

        # Set and display confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(targets, predictions, labels=labels, normalize='all'), 
            display_labels=display_labels
            )
        disp.plot(xticks_rotation='vertical', include_values=False)