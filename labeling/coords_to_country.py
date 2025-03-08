import pandas as pd # remember to install package "pandas"
from geopy.geocoders import Nominatim # remember to install package "geopy"
from geopy.extra.rate_limiter import RateLimiter

#Load the csv file
datafile = pd.read_csv('../coordinates.csv')

# Initialize Nominatim geocoder (Set to be unique and descriptive)
geocoder = Nominatim(user_agent="GeoguessrAI (yuhac30@uci.edu)")

# Pause 1.5 seconds between requests, to prevent exceeding usage limit of Nominatim.
# On my device if I set the time limit to below 1 second it will instantly raise 403 Forbidden error,
# but feel free to experiment with lower time limits!
reverse_geocoder = RateLimiter(geocoder.reverse, min_delay_seconds=1.5)

def get_country(latitude, longitude):
    """
    Takes in two parameters latitude and longitude (type numpy.float64), returns country name (type string)
    """
    lat = float(latitude)
    lon = float(longitude)

    location = reverse_geocoder((lat, lon), exactly_one=True, language="en")
    if location and location.raw.get('address'):
        return location.raw['address'].get('country', 'Unknown')
    return 'Unknown'

#print(get_country(20.82488495, -98.49951688)) #for testing

# Process rows in chunks of 10 and save progress
CHUNK_SIZE = 10
for start in range(20890, len(datafile), CHUNK_SIZE):
    """ 
    Change the starting condition of the for loop to run on different data 
    (Eg. "for start in range(200, len(datafile), CHUNK_SIZE)" generates information for images 200~209)
    """

    end = min(start + CHUNK_SIZE, len(datafile))  # Ensure we don't go out of bounds
    for i in range(start, end):
        row = datafile.iloc[i]
        datafile.at[i, 'country'] = get_country(row['latitude'], row['longitude'])

    # Save progress every CHUNK_SIZE rows
    datafile.iloc[start:end].to_csv(f'countries\\coordinates_with_countries_chunk_{start}_{end}.csv',
                                    index = False)

# Save the updated file to a new CSV
#datafile.to_csv('coordinates_with_countries.csv', index=False)