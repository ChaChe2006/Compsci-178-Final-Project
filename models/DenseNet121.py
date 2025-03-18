import tensorflow as tf

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
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file containing data paths and corresponding labels
df = pd.read_csv('dataset/train/image_paths_with_country.csv')
print(df.head())

def group_countries(df, min_samples=100):
    country_counts = df['country'].value_counts()
    keep_countries = country_counts[country_counts >= min_samples].index
    df['country_grouped'] = df['country'].apply(
        lambda x: x if x in keep_countries else 'Other'
    )
    return df

df = group_countries(df, min_samples=100)

def balanced_sampling(df, target_samples=1000, random_state=42):
    class_counts = df['country_grouped'].value_counts()
    balanced_dfs = []
    for country, count in class_counts.items():
        class_df = df[df['country_grouped'] == country]
        if count < target_samples:
            oversampled = class_df.sample(
                n=target_samples - count,
                replace=True,
                random_state=random_state
            )
            balanced_dfs.append(pd.concat([class_df, oversampled]))
        else:
            balanced_dfs.append(class_df.sample(
                n=target_samples,
                random_state=random_state
            ))
    return pd.concat(balanced_dfs).sample(frac=1, random_state=random_state)

df = balanced_sampling(df, target_samples=1000)

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.densenet.preprocess_input(img_array)
    return img_array

label_encoder = LabelEncoder()
df['class_label_encoded'] = label_encoder.fit_transform(df['country_grouped'])
num_classes = len(label_encoder.classes_)
df['class_label_onehot'] = df['class_label_encoded'].apply(lambda x: to_categorical(x, num_classes))
print(df.head())

def create_dataset(df, image_dir, batch_size=32, target_size=(224, 224)):
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
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
image_dir = 'dataset/train'
batch_size = 32
image_size = (224, 224)
train_dataset = create_dataset(train_df, image_dir, batch_size, image_size).repeat()
val_dataset = create_dataset(val_df, image_dir, batch_size, image_size).repeat()

# Choose DenseNet model variant (DenseNet121, DenseNet169, or DenseNet201)
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)  # Added Batch Normalization
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)  # Added Batch Normalization
x = Dropout(0.3)(x)

predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=1e-4, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    steps_per_epoch=len(train_df) // batch_size,
    validation_steps=len(val_df) // batch_size,
    callbacks=[lr_scheduler, early_stopping]
)

model.save('CountryClassifier_DenseNet121_10_epochs.keras')

# Load test data from CSV
test_df = pd.read_csv('dataset/test/test_image_paths_with_country.csv')

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

# Sample predictions on the first few batches
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

# Generate predictions for all test samples
y_pred = []
y_true = []

for images, labels in test_dataset:
    batch_preds = model.predict(images)
    batch_preds_classes = np.argmax(batch_preds, axis=1)
    batch_true_classes = np.argmax(labels.numpy(), axis=1)
    
    y_pred.extend(batch_preds_classes)
    y_true.extend(batch_true_classes)

# Convert to numpy arrays
y_pred = np.array(y_pred)
y_true = np.array(y_true)

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, labels=np.arange(len(label_encoder.classes_))))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Save the Model
model.save('CountryClassifier_DenseNet121_10_epochs.keras')

# Plot training history
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('Densenet121_training_history.png')
plt.show()

# Additional analysis - plot top N classes by accuracy
def plot_class_accuracy(y_true, y_pred, class_names, top_n=10):
    """Plot the top N classes by accuracy"""
    accuracies = {}
    for i, class_name in enumerate(class_names):
        # Calculate accuracy for this class
        class_indices = np.where(np.array(y_true) == i)[0]
        if len(class_indices) > 0:
            correct = sum(np.array(y_pred)[class_indices] == i)
            accuracies[class_name] = correct / len(class_indices)
    
    # Sort by accuracy
    sorted_classes = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Plot
    plt.figure(figsize=(12, 6))
    names, accs = zip(*sorted_classes)
    plt.bar(names, accs)
    plt.title(f'Top {top_n} Classes by Accuracy')
    plt.xlabel('Country')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('Densenet121_top_classes.png')
    plt.show()

# Plot top classes
plot_class_accuracy(y_true, y_pred, label_encoder.classes_, top_n=10)