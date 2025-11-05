import random
import csv

# Define the range for X and Y coordinates
min_coord = 0
max_coord = 1000

# Create a set to store unique points
unique_points = set()

# Generate 100 unique X,Y points
while len(unique_points) < 100:
    x = random.randint(min_coord, max_coord)
    y = random.randint(min_coord, max_coord)
    unique_points.add((x, y))

# Define the CSV file name
csv_filename = input("Enter the name of the CSV file (e.g., data.csv): ")

# Write the unique points to the CSV file
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['X', 'Y'])  # Write header row
    for point in unique_points:
        csv_writer.writerow(list(point))

print(f"Generated 100 unique X,Y points and saved to '{csv_filename}'")