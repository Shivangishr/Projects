

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, MaxPooling1D, Dropout, LSTM, Bidirectional, SpatialDropout1D, TextVectorization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

import os

train_data = pd.read_csv('/content/drive/MyDrive/toxic_comments/train.csv',index_col='id')
train_data.head()

train_data['comment_length'] = train_data['comment_text'].apply(lambda row: len(row))
train_data.head()

import seaborn as sns
sns.displot(
    data=train_data,
    x="comment_length",
    hue='toxic',
    multiple="stack",
)

toxic_data = train_data[train_data['toxic'] == 1]
toxic_data.head()

print(
    'max', toxic_data['comment_length'].max(),
    'min', toxic_data['comment_length'].min(),
    'mean', toxic_data['comment_length'].mean(),
    'median', toxic_data['comment_length'].median(),
    '75%', toxic_data['comment_length'].quantile(0.75),
)

max_comment_len = 300

test_data = pd.read_csv('/content/drive/MyDrive/toxic_comments/test.csv',index_col='id')
test_data.head()

test_labels_data = pd.read_csv('/content/drive/MyDrive/toxic_comments/test_labels.csv',index_col='id')
test_labels_data.head()

test_data = test_data.join(test_labels_data)
test_data = test_data[test_data['toxic'] != -1]
test_data.head()

num_words = 10000
encoder = TextVectorization(max_tokens=num_words)
encoder.adapt(train_data['comment_text'].values)

model_lstm = Sequential([
    encoder,
    Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=128,
        mask_zero=True,
        input_length=max_comment_len,
    ),
    SpatialDropout1D(0.5),
    LSTM(40, return_sequences=True),
    LSTM(40),
    Dense(6, activation='sigmoid'),
])

model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

model_lstm_save_path = '/content/drive/MyDrive/toxic_comments/model'
checkpoint_callback_lstm = ModelCheckpoint(
    model_lstm_save_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1,
    save_format='tf',
)

x_train = train_data['comment_text'].values
y_train = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

x_train[:2]

y_train[:2]

history_lstm = model_lstm.fit(
    x_train,
    y_train,
    epochs=15, # 15
    batch_size=512,
    validation_split=0.2,
    callbacks=[checkpoint_callback_lstm],
)

plt.plot(history_lstm.history['accuracy'],
         label='Training accuracy')
plt.plot(history_lstm.history['val_accuracy'],
         label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

x_test = test_data['comment_text'].values
y_test = test_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

model_lstm.evaluate(x_test, y_test, verbose=1)

saved_model = load_model(model_lstm_save_path)
saved_model.evaluate(x_test, y_test, verbose=1)

test_data[test_data['toxic']==1][:5]

test_data[test_data['toxic']==1][:5]['comment_text'].values

test_labels = saved_model.predict(test_data[test_data['toxic']==1][:5]['comment_text'].values)
for labels in test_labels:
    print([ round(lbl, 2) for lbl in labels])

quotes = [
    "Love is a kind laguage. You wont understand. Monkey! ",

]
not_toxic_prediction = saved_model.predict(quotes)
for labels in not_toxic_prediction:
    print([ round(lbl, 2) for lbl in labels])
