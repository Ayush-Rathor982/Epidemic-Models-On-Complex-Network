import pandas as pd

# Load the original CSV file
df = pd.read_csv("owid-covid-data-old.csv")  # Replace with your file name

# Filter for rows where location is Afghanistan
afghanistan_data = df[df['location'] == 'Japan']

# Select only the columns you want
filtered_data = afghanistan_data[['location', 'date', 'new_cases', 'new_deaths', 'new_cases_per_million', 'new_deaths_per_million', 'reproduction_rate', 'people_vaccinated']]

# Save to new CSV file
filtered_data.to_csv("japan_cases.csv", index=False)
