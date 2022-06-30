from pandas import Index
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# Εισαγωγή δεδομένων απο το excel
data = pd.read_excel('test_data.xlsx', 'Data', usecols='A:E')

# Data info
data.shape
data.info()
df = pd.DataFrame(columns=['Microservices', 'Type', 'Pass', 'Failure', 'Target'])
data.columns = Index(['Microservices', 'Type', 'Pass', 'Failure', 'Target'], dtype=object)

# Define Data
data.Failure.plot(kind="hist", figsize=(10, 6))
data.Failure = np.where(data.Failure == 0, 0, 1)
data.Failure.value_counts()
data.info()
data.Pass.plot(kind="hist", figsize=(10, 6))

print("Test finally pass or fail but most of scenario had the opposite result")
df = data.groupby(['Type', 'Target']).Pass.nunique().reset_index()
df1 = data.groupby(['Type', 'Target']).count()[['Pass']]
df1.plot(figsize=(30, 10), kind='bar', color='r')
print(df1)

# Create X_train, y_train, X_test, y_test
X = data.drop(["Target", "Microservices", "Type"], axis=1)
y = data["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)

# Separate number data
data1 = data.drop(["Microservices", "Type"], axis=1)
dataset = data1.copy()
dataset.tail()
dataset.isna().sum()
dataset = dataset.dropna()

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_dataset.describe().transpose()
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Failure')
test_labels = test_features.pop('Failure')

# Data Normalization
TargetClass = np.array(train_features['Target'])
Failures_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
Failures_normalizer.adapt(TargetClass)
Failures_model = tf.keras.Sequential([
    Failures_normalizer,
    layers.Dense(units=1)
])

# Create a model and train it
Failures_model.summary()
Failures_model.predict(TargetClass[:10])
Failures_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = Failures_model.fit(
    train_features['Target'],
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [Failure]')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_loss(history)

test_results = {}
test_results['Failures_model'] = Failures_model.evaluate(
    test_features['Target'],
    test_labels, verbose=0)
x = tf.linspace(0.0, 250, 251)
y = Failures_model.predict(x)


def plot_Target(x, y):
    plt.scatter(train_features['Target'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Target')
    plt.ylabel('Failure')
    plt.legend()
    plt.show()


plot_Target(x, y)


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(500, activation='relu'),
        layers.Dense(1)

    ])
    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


dnn_Failures_model = build_and_compile_model(Failures_normalizer)
dnn_Failures_model.summary()
history = dnn_Failures_model.fit(
    train_features['Target'],
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

plot_loss(history)
x = tf.linspace(0.0, 250, 251)
y = dnn_Failures_model.predict(x)
plot_Target(x, y)

test_results['dnn_Failures_model'] = dnn_Failures_model.evaluate(
    test_features['Target'], test_labels,
    verbose=0)
