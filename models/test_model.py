import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os


def load_model_and_encoder(model_path, label_list):
    # Load the trained model
    model = load_model(model_path)

    # Recreate the LabelEncoder and fit it using the list of countries from training data
    label_encoder = LabelEncoder()
    label_encoder.fit(label_list)  # fit it to the same classes you used during training

    return model, label_encoder


def load_and_preprocess_image(image_path, target_size = (224, 224)):
    """
    Load an image, resize it, and normalize pixel values.
    """
    img = image.load_img(image_path, target_size = target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
    return img_array


def predict_country(model, label_encoder, image_path):
    # Preprocess the image
    img_array = load_and_preprocess_image(image_path)

    # Add an extra dimension (batch size of 1)
    img_array = np.expand_dims(img_array, axis = 0)

    # Make the prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis = 1)

    # Convert the predicted class index to the country name
    predicted_country = label_encoder.inverse_transform(predicted_class)
    return predicted_country[0]


def main(image_path, model_path = 'CountryClassifier_EfficientNetB0_20_epochs.keras',
         label_list = None):
    if label_list is None:
        label_list = ['United States', 'Brazil', 'Australia', 'Argentina', 'Russia', 'France',
                      'Nigeria',
                      'Poland', 'United Kingdom', 'Turkey', 'Portugal', 'Japan', 'Mexico', 'India',
                      'Italy',
                      'South Korea', 'Malaysia', 'Finland', 'Sweden', 'Colombia', 'Thailand',
                      'Indonesia',
                      'Ireland', 'Spain', 'Czechia', 'Austria', 'South Africa', 'Sri Lanka',
                      'Kenya', 'Netherlands',
                      'Mongolia', 'Switzerland', 'Israel', 'Romania', 'Belgium', 'Peru', 'Botswana',
                      'Germany',
                      'Bulgaria', 'Chile', 'Slovakia', 'Senegal', 'Ukraine', 'Lithuania',
                      'Philippines', 'Denmark',
                      'Canada', 'Ghana', 'Bangladesh', 'Uruguay', 'Cambodia', 'Bolivia', 'Croatia',
                      'Singapore',
                      'Serbia', 'Mali', 'Palestinian Territory', 'China', 'Ecuador', 'Hungary',
                      'Greenland', 'Greece',
                      'Latvia', 'Estonia', 'Tunisia', 'Guatemala', 'New Zealand', 'Montenegro',
                      'Kyrgyzstan', 'Slovenia',
                      'North Macedonia', 'Panama', 'Pakistan', 'Norway', 'Iceland', 'Lesotho',
                      'Albania', 'Bhutan',
                      'Eswatini', 'Malta', 'Armenia', 'Iran', 'Madagascar', 'Jordan', 'Costa Rica',
                      'Taiwan', 'Tanzania',
                      'Trinidad and Tobago', 'Belarus', 'Andorra', 'Kazakhstan', 'Georgia',
                      'El Salvador', 'United Arab Emirates',
                      'Uganda', 'Dominican Republic', 'Morocco', 'Guinea-Bissau', 'Luxembourg',
                      'Algeria', 'Venezuela',
                      'Cuba', 'Sierra Leone', 'Uzbekistan', 'Vanuatu', 'Vietnam', 'Saudi Arabia',
                      'Egypt', 'Côte d\'Ivoire',
                                             'Iraq', 'Ethiopia', 'The Gambia', 'East Timor',
                      'Kosovo', 'Namibia', 'Syria', 'Papua New Guinea',
                      'Guinea', 'Honduras', 'Myanmar', 'Congo-Brazzaville', 'Rwanda']

    # Load model and label encoder
    model, label_encoder = load_model_and_encoder(model_path, label_list)

    # Make prediction
    predicted_country = predict_country(model, label_encoder, image_path)

    print(f"The predicted country is: {predicted_country}")


if __name__ == '__main__':
    image_path = input("Enter image file path:")
    main(image_path)
