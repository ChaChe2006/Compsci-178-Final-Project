import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

#Load the csv file
datafile = pd.read_csv('coordinates.csv')

# Initialize Nominatim geocoder
geocoder = Nominatim(user_agent="geoapiExercises")

# Pause 1 second between requests, to prevent exceeding usage limit of Nominatim
reverse_geocoder = RateLimiter(geocoder.reverse, min_delay_seconds=1)

def get_country(latitude, longitude):
    """
    Takes in two parameters latitude and longitude (type double), returns country name (type string)
    """
    location = reverse_geocoder((latitude, longitude), exactly_one=True, language="en")
    if location and location.raw.get('address'):
        return location.raw['address'].get('country', 'Unknown')
    return 'Unknown'

print(get_country(20.82488495, -98.49951688))

# Apply the function to each row of our datafile
#for index, row in datafile.iterrows():
#    datafile.at[index, 'country'] = get_country(row['latitude'], row['longitude'])

# Save the updated file to a new CSV
#datafile.to_csv('coordinates_with_countries.csv', index=False)