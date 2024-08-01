import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from transformers import DistilBertTokenizer, TFAutoModel
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from tensorflow.keras import backend as K # type: ignore


TRAIN_CSV_CLEAN = "csv/clean/train.csv"
PREDICT_CSV_CLEAN = "csv/clean/predict.csv"


SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

CHECKPOINT = 'distilbert-base-uncased'
NUM_LABELS = 3  # Number of classes (negative, neutral, positive)
BATCH_SIZE = 32
EPOCHS = 10


class ClassifModel(tf.keras.Model):

    def __init__(self, checkpoint, num_labels):
        super(ClassifModel, self).__init__()
        
        self.base_model = TFAutoModel.from_pretrained(checkpoint)
        self.base_model.trainable = False  # Freeze the outer model
        
        self.pooler = tf.keras.layers.Lambda(lambda x: x[:, 0, :])  # Use the [CLS] token's representation
        self.flatten = tf.keras.layers.Flatten()
        self.dropout1 = tf.keras.layers.Dropout(rate = 0.25)
        self.linear1 = tf.keras.layers.Dense(units=1024, kernel_regularizer=tf.keras.regularizers.l1_l2(0.01))
        self.batchNorm1 = tf.keras.layers.BatchNormalization()
        self.activation1 = tf.keras.layers.Activation("relu")        
        self.out = tf.keras.layers.Dense(units=num_labels, activation="softmax")  # Multi-class classification        

    def call(self, inputs, training = False):
        x = self.base_model(inputs)[0]  # [0] gets the last_hidden_state
        x = self.pooler(x) # Pooling layer to get [CLS] token's representation
        x = self.flatten(x)
        x = self.dropout1(x) if training else x # ensure dropout is only applied during training
        x = self.linear1(x)
        x = self.batchNorm1(x)
        x = self.activation1(x)
        x = self.out(x)
        return x

def f1_score(y_true, y_pred):
    y_pred_classes = K.argmax(y_pred, axis=-1)  # Get the index of the max value
    y_true = K.cast(y_true, 'int64')
    
    # Convert y_pred and y_true to one-hot format
    y_pred_classes = K.one_hot(y_pred_classes, num_classes=3)
    y_true = K.one_hot(y_true, num_classes=3)
    
    tp = K.sum(K.cast(y_true * y_pred_classes, 'float32'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred_classes, 'float32'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred_classes), 'float32'), axis=0)
    
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return K.mean(f1)


def preprocess_data(texts, labels=None, max_len=128):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors='tf')
    if labels is not None:
        labels = tf.convert_to_tensor(labels)
    return encodings, labels


def create_dataset(texts, labels=None, batch_size=BATCH_SIZE):
    encodings, labels = preprocess_data(texts, labels)
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(dict(encodings))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
    return dataset


if __name__ == "__main__":
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = ClassifModel(CHECKPOINT, NUM_LABELS)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy', f1_score]
    )

    df = pd.read_csv(TRAIN_CSV_CLEAN)

    df_train, df_temp = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df['label'])
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=SEED, stratify=df_temp['label'])

    train_dataset = create_dataset(df_train['tweet'].tolist(), df_train['label'].tolist())
    val_dataset = create_dataset(df_val['tweet'].tolist(), df_val['label'].tolist())
    test_labels = df_test['label'].tolist()
    test_dataset = create_dataset(df_test['tweet'].tolist(), test_labels)
    
    print(f'Split trainign data ({TRAIN_CSV_CLEAN}) - {len(df)}:')
    print(f"Train: {len(list(train_dataset))}, Val: {len(list(val_dataset))}, Test: {len(list(test_dataset))}")

    # Monitor val_f1_score and stop after 3 epochs without improvement
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_f1_score', patience=3, mode='max', restore_best_weights=True
    )
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[earlystop]
    )

    # Evaluate the model
    loss, accuracy, f1 = model.evaluate(test_dataset)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')
    print(f'Test F1: {f1}')

    predictions = model.predict(test_dataset)
    predicted_labels = np.argmax(predictions, axis=-1) # Post-process predictions 
    test_labels_np = np.array(test_labels)

    # Compute confusion matrix and classification report
    cm = confusion_matrix(test_labels_np, predicted_labels)
    report = classification_report(
        test_labels_np, predicted_labels, 
        target_names=['Against', 'None', 'Favor'], 
        zero_division=1 # 0 is also possible
    )

    print(f'Confusion Matrix:\n{cm}')
    print(f'\nClassification Report:\n{report}')

    df_pred = pd.read_csv(PREDICT_CSV_CLEAN, low_memory=False)
    tweets_to_predict = df_pred['tweet'].tolist()
    pred_dataset = create_dataset(tweets_to_predict, labels=None, batch_size=BATCH_SIZE)

    print(f'start predicting {len(tweets_to_predict)} tweets...')

    predictions = []
    for batch in tqdm(pred_dataset, total=len(tweets_to_predict) // BATCH_SIZE, desc="Predicting"):
        batch_predictions = model.predict(batch)
        predictions.extend(batch_predictions)

    predicted_labels = np.argmax(predictions, axis=-1) # Convert predictions to labels

    stance_results_csv = 'stance.csv'
    print(f'*** Predicitng done! ***\Saving to {stance_results_csv}')
    stance_df = pd.DataFrame({
        'raw_predictions': list(predictions),
        'predicted_label': predicted_labels,
        'tweet': tweets_to_predict,
    })
    stance_df.to_csv(stance_results_csv, index=False)
