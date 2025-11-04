import csv

def print_csv_data():
    # Prompts the user for a CSV filename and prints its contents.
    filename = input("Enter the name of the CSV file (e.g., data.csv): ")

    try:
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                print(', '.join(row)) # Prints each row with elements separated by a comma and space
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    print_csv_data()