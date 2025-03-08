def print_stats(filename):
    """
    Organize countries and the number of times they appear in a dict.
    Print out the dict in descending order based on the number of appearances.
    """

    country_counts = {}

    with open(filename, 'r') as file:

        next(file) # Skip the first line (header)

        # Process remaining lines
        for line in file:
            country = line.split(',')[2]
            if country in country_counts:
                country_counts[country] += 1
            else:
                country_counts[country] = 1

    # sort by value
    sorted_countries = sorted(country_counts.items(), key = lambda x: x[1], reverse = True)

    print("Countries and the number of times they appear in the dataset:")
    for country, count in sorted_countries:
        print(f"{country}: {count}")

print_stats("coordinates_with_country.csv")