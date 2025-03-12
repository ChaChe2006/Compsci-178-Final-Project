import tensorflow as tf
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.__version__)
# run on GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("GPU is enabled!")
    except RuntimeError as e:
        print(e)

import pandas as pd
import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file containing data paths and corresponding labels
df = pd.read_csv('../dataset/train/image_paths_with_country.csv')
print(df.head())


def group_countries(df, min_samples = 100):
    """
    Group countries with less than min_samples into 'Other' category
    """
    # Get country counts
    country_counts = df['country'].value_counts()

    # Create mapping for countries to keep
    keep_countries = country_counts[country_counts >= min_samples].index

    # Apply grouping
    df['country_grouped'] = df['country'].apply(
        lambda x: x if x in keep_countries else 'Other'
    )

    return df

# Apply grouping
df = group_countries(df, min_samples = 100)


def balanced_sampling(df, target_samples = 1000, random_state = 42):
    """
    Balance dataset through oversampling and undersampling
    """
    # Separate classes
    class_counts = df['country_grouped'].value_counts()

    balanced_dfs = []
    for country, count in class_counts.items():
        class_df = df[df['country_grouped'] == country]

        if count < target_samples:
            # Oversample minority classes
            oversampled = class_df.sample(
                n = target_samples - count,
                replace = True,
                random_state = random_state
            )
            balanced_dfs.append(pd.concat([class_df, oversampled]))
        else:
            # Undersample majority classes
            balanced_dfs.append(class_df.sample(
                n = target_samples,
                random_state = random_state
            ))

    return pd.concat(balanced_dfs).sample(frac = 1, random_state = random_state)


# Apply balanced sampling
df = balanced_sampling(df, target_samples = 1000)

def load_and_preprocess_image(image_path, target_size=(456, 456)):
    """
    Load an image, resize it, and normalize pixel values.
    """
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
    return img_array

# Encode class labels as integers
label_encoder = LabelEncoder()
df['class_label_encoded'] = label_encoder.fit_transform(df['country'])

# Convert to one-hot encoded vectors
num_classes = len(label_encoder.classes_)
df['class_label_onehot'] = df['class_label_encoded'].apply(lambda x: to_categorical(x, num_classes))

print(df.head())

def create_dataset(df, image_dir, batch_size=32, target_size=(456, 456)):
    """
    Creates a TensorFlow Dataset.
    """
    def generator():
        for _, row in df.iterrows():
            image_path = os.path.join(image_dir, row['image_path'])
            img_array = load_and_preprocess_image(image_path, target_size)
            label = row['class_label_onehot']
            yield img_array, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(*target_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
        )
    )

    # Batch and prefetch for performance
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Create datasets
image_dir = 'dataset/train'
batch_size = 32
image_size = (456, 456)
train_dataset = create_dataset(train_df, image_dir, batch_size, image_size).repeat() #make sure dataset repeats indefinitely to prevent out of bounds error
val_dataset = create_dataset(val_df, image_dir, batch_size, image_size).repeat()

base_model = EfficientNetB5(weights='imagenet', include_top=False, input_shape=(456, 456, 3))
base_model.trainable = True
for layer in base_model.layers[:300]:  # Unfreeze the top 100 layers
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)  # Add dropout for regularization
x = Dense(512, activation='relu')(x) # Add another dense layer for more complexity

predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Nadam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_callback = ModelCheckpoint(
    'model_checkpoint.keras',  # Path where the model will be saved
    monitor='val_loss',  # Metric to monitor (e.g., 'val_loss' or 'val_accuracy')
    save_best_only=False,  # Save only the best model (based on validation loss or accuracy)
    save_weights_only=False,  # Save the full model (not just weights)
    mode='min',  # Save the model when the monitored metric is minimized (e.g., 'val_loss')
    verbose=1  # Print a message when saving the model
)

if os.path.exists("model_checkpoint.keras"):
    print("Loading checkpoint...")
    model = tf.keras.models.load_model("model_checkpoint.keras")

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    steps_per_epoch = len(train_df) // batch_size,
    validation_steps = len(val_df) // batch_size,
    callbacks=[checkpoint_callback]
)

# Load test data from CSV
test_df = pd.read_csv('../dataset/test/test_image_paths_with_country.csv')

# Apply the same grouping as training data
test_df = group_countries(test_df, min_samples=100)

# Filter out classes that don't exist in the training data
test_df = test_df[test_df['country_grouped'].isin(label_encoder.classes_)]

# Encode using the same label encoder
test_df['class_label_encoded'] = label_encoder.transform(test_df['country_grouped'])
test_df['class_label_onehot'] = test_df['class_label_encoded'].apply(
    lambda x: to_categorical(x, num_classes).tolist()
)

# Create test dataset
test_dataset = create_dataset(test_df, image_dir, batch_size, image_size)

for images, labels in test_dataset.take(5):  # Take one batch
    preds = model.predict(images)
    actual_labels = np.argmax(labels.numpy(), axis=1)  # Convert to class indices

    # Convert indices to country names
    predicted_countries = label_encoder.inverse_transform(preds.argmax(axis=1))  # Convert predicted indices to country names
    actual_countries = label_encoder.inverse_transform(actual_labels)  # Convert actual indices to country names

    # Print out the results
    print("Predicted:", predicted_countries)
    print("Actual:", actual_countries)

# Evaluate on test data
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Generate predictions
y_pred = model.predict(test_dataset)
y_pred = np.argmax(y_pred, axis=1)
y_true = test_df['class_label_encoded'].values  # Use encoded labels from the test CSV

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_df['country'].unique(), labels=np.arange(len(test_df['country'].unique()))))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Save the Model
### Remember to change the model name to prevent overwriting existing models ###
model.save('CountryClassifier_EfficientNetB5_10_epochs.keras')

# Plot training history
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
