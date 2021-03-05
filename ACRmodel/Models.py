#!/usr/bin/env python3
import mir_eval
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
import tensorflow
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing



class MLP():
    """
    Very simple MLP model with one 100 units layer.
    """
    def __init__(self, max_iter=500, random_state=1):
        self.model = MLPClassifier(max_iter=max_iter, random_state=random_state)
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



class MLP_scalered(MLP):
    def __init__(self, max_iter=500, random_state=1):
        scalered_mlp = sklearn.pipeline.Pipeline([
          ('scaler', sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)),
          ('estimator', MLPClassifier(max_iter=max_iter, random_state=random_state))
        ])
        self.model = scalered_mlp
        print("[INFO] The MLP model with scaler preprocessing was successfully created.")





class CRNN():
    """
    Very basic CRNN model, maybe not working, who knows.
    """
    def __init__(self, input_shape, output_classes):
        n_frames, n_chromas, chanells = input_shape
        # Create model
        model = tensorflow.keras.models.Sequential()

        # Feature Extractor
        model.add(tensorflow.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape,padding='same'))
        model.add(tensorflow.keras.layers.MaxPooling2D((2,2),padding='same'))
        model.add(tensorflow.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'))
        model.add(tensorflow.keras.layers.MaxPooling2D((2,2),padding='same'))
        model.add(tensorflow.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'))

        # Classifier - RNN
        model.add(tensorflow.keras.layers.Flatten())
        model.add(tensorflow.keras.layers.Dense(64, activation='relu'))
        model.add(tensorflow.keras.layers.Dense(output_classes, activation='softmax'))

        # Compile model
        model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.1),
            loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )


        model.summary()
        self.model = model
        print("[INFO] The CRNN model was successfully created.")

    def fit(self, data, targets, dev_data, dev_targets, epochs=50):
        # Train model
        self.history = self.model.fit(
            data, targets, epochs=epochs,
            validation_data=(dev_data, dev_targets)
        )
        print("[INFO] The CRNN model was successfully trained.")

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

        # Generate predictions and targets
        predictions = self.model.predict(data)
        a1, a2, a3 = predictions.shape
        predictions = predictions.reshape((a1*a2, a3))
        predictions = tensorflow.argmax(predictions, axis=1)
        a1, a2 = targets.shape
        targets = targets.reshape((a1*a2))

        # Set and display confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(targets, predictions, labels=labels, normalize='all'), 
            display_labels=display_labels
            )
        disp.plot(xticks_rotation='vertical', include_values=False)



class CRNN_1(CRNN):
    """
    CRNN model inspired by Junyan Jiang, Ke Chen, Wei li, Gus Xia, 2019.
    """
    def __init__(self, input_shape, output_classes):
        n_frames, n_chromas, chanells = input_shape
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
        model.add(tensorflow.keras.layers.Reshape((a1, a2*a3), input_shape=(n_frames, a2, a3)))
        model.add(tensorflow.keras.layers.Bidirectional(
            tensorflow.keras.layers.LSTM(96, return_sequences=True))
        )
        model.add(
            tensorflow.keras.layers.Dense(output_classes, activation='softmax')
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



class CRNN_2(CRNN):
    """
    CRNN model inspired by Brian McFee and Juan Pablo Bello, 2017.
    """
    def __init__(self, input_shape, output_classes):
        n_frames, n_chromas, chanells = input_shape
        # Create model
        model = tensorflow.keras.models.Sequential()

        # Feature Extractor
        model.add(tensorflow.keras.layers.Conv2D(1, (5,5), activation='relu', input_shape=input_shape,padding='same'))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.Reshape((n_frames, n_chromas)))
        model.add(tensorflow.keras.layers.Conv1D(36, kernel_size=1))


        # Classifier - RNN
        model.add(tensorflow.keras.layers.Bidirectional(
            tensorflow.keras.layers.GRU(256, return_sequences=True))
        )
        model.add(
            tensorflow.keras.layers.Dense(output_classes, activation='softmax')
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



class MLP2RNN():
    """
    sklearn scalered MLP -> tensorflow RNN
    """
    def __init__(self, input_shape, output_classes, max_iter=500, random_state=7):
        n_frames, n_chromas, chanells = input_shape
        # ACOUSTIC model

        # Create Pipeline
        scalered_mlp = sklearn.pipeline.Pipeline([
          ('scaler', sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)),
          ('estimator', MLPClassifier(max_iter=max_iter, random_state=random_state))
        ])

        self._acoustic_model = scalered_mlp
        print("[INFO] The MLP model with scaler preprocessing was successfully created.")




        # LINGUISTIC model

        # Create model
        model = tensorflow.keras.models.Sequential()

        model.add(tensorflow.keras.layers.Bidirectional(
            tensorflow.keras.layers.LSTM(256, return_sequences=True))
        )
        model.add(
            tensorflow.keras.layers.Dense(output_classes, activation='softmax')
        )

        # Compile model
        model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(),
            loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        model.summary()
        self._linguistic_model = model
        print("[INFO] The RNN model was successfully created.")


    def fit(self, data, targets, epochs=50):
        # Train ACOUSTIC model

        # Preprocess acoustic data
        acoustic_data, acoustic_targets, dev_acoustic_data, dev_acousting_targets = MLP2RNN.preprocess_acoustic(data, targets)
        # Fit model
        self._acoustic_model.fit(acoustic_data, acoustic_targets)
        print("[INFO] The acoustic model was successfully trained with dev accuracy {:.2f}".format(100*self._acoustic_model.score(data, targets)), "%")
        # Display results
        self.display_acoustic_confusion_matrix(dev_acoustic_data, dev_acousting_targets)




        # Train LINGUISTIC model

        # Preprocess linguistic data
        linguistic_data, linguistic_target, dev_linguistic_data, dev_linguistic_targets = MLP2RNN.preprocess_linguistic(data, targets)
        # Fit model
        self._linguistic_history = self._linguistic_model.fit(
            linguistic_data, linguistic_target, epochs=epochs,
            validation_data=(dev_linguistic_data, dev_linguistic_targets)
        )
        print("[INFO] The linguistic model was successfully trained with dev accuracy {:.2f}".format(100*self._acoustic_model.evaluate(data, targets, verbose=2)[1]), "%")
        # Display results
        self.display_linguistic_confusion_matrix(dev_acoustic_data, dev_acousting_targets)
        self.display_linguistic_training_progress()



    def preprocess_acoustic(self, data, targets):
        window_size = 5
        test_size = 0.3
        random_state = None
        acoustic_data, acoustic_targets = [], []

        # Preprocess data
        _, n_frames, n_features = data.shape
        for sequence in data:
            for i in range(n_frames):
                acoustic_data.append(
                    np.concatenate((
                        np.zeros((abs(min(0, i-window_size)), n_features)),
                        np.array(sequence[:, max(0, i-window_size):min(i+window_size+1, n_frames)]),
                        np.zeros((abs(min(0, (n_frames)-(i+window_size+1))), n_features))
                    ), axis = 0).flatten()
                )
        # Preprocess targets
        n_sequences, n_frames = targets.shape
        acoustic_targets = targets.reshape((n_sequences*n_frames))

        # Divide dataset to training and developing sets
        train_x, dev_x, train_y, dev_y = sklearn.model_selection.train_test_split(np.array(acoustic_data), np.array(acoustic_targets), test_size=test_size, random_state=random_state)
        return train_x, dev_x, train_y, dev_y

    def preprocess_linguistic(self, data, targets):
        test_size = 0.3
        random_state = None
        linguistic_data, linguistic_targets = [], targets

        # Preprocess data
        for sequence in data:
            linguistic_data.append(self._acoustic_model.predict_proba(sequence))


        train_x, dev_x, train_y, dev_y = sklearn.model_selection.train_test_split(np.array(linguistic_data), np.array(linguistic_targets), test_size=test_size, random_state=random_state)
        return train_x, dev_x, train_y, dev_y



    def display_linguistic_training_progress(self):
        plt.plot(self._linguistic_history.history['accuracy'], label='accuracy')
        plt.plot(self._linguistic_history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.0, 1])
        plt.legend(loc='lower right')
        plt.show()

    def display_acoustic_confusion_matrix(self, data, targets):
        # Define labels
        display_labels = np.array(["N", "C", "C:min", "C#", "C#:min", "D", "D:min", "D#", "D#:min", "E", "E:min", "F", "F:min", "F#", "F#:min", "G", "G:min", "G#", "G#:min", "A", "A:min", "A#", "A#:min", "B", "B:min"])
        labels = np.array([i for i in range(len(display_labels))])

        # Generate predictions
        predictions = self._acoustic_model.predict(data)

        # Set and display confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(targets, predictions, labels=labels, normalize='all'),
            display_labels=display_labels
            )
        disp.plot(xticks_rotation='vertical', include_values=False)

    def display_linguistic_confusion_matrix(self, data, targets):
        # Define labels
        display_labels = np.array(["N", "C", "C:min", "C#", "C#:min", "D", "D:min", "D#", "D#:min", "E", "E:min", "F", "F:min", "F#", "F#:min", "G", "G:min", "G#", "G#:min", "A", "A:min", "A#", "A#:min", "B", "B:min"])
        labels = np.array([i for i in range(len(display_labels))])

        # Generate predictions and targets
        predictions = self._linguistic_history.predict(data)
        a1, a2, a3 = predictions.shape
        predictions = predictions.reshape((a1*a2, a3))
        predictions = tensorflow.argmax(predictions, axis=1)
        a1, a2 = targets.shape
        targets = targets.reshape((a1*a2))

        # Set and display confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(targets, predictions, labels=labels, normalize='all'),
            display_labels=display_labels
            )
        disp.plot(xticks_rotation='vertical', include_values=False)
