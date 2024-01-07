#PRIMER
import pandas as pd


def format_csv_column_names(input_csv_path, output_csv_path, decimal_places=5):
    # Read the CSV file
    df = pd.read_csv(input_csv_path)

    # Replace underscores with spaces in the column names
    df.columns = [col.replace('_', ' ') for col in df.columns]

 	# Format all float columns to round to the specified number of decimal places
    for col in df.select_dtypes(include=['float']):
        df[col] = df[col].round(decimal_places)

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv_path, index=False)


# Define the path to your CSV file and the output LaTeX file
csv_file_path = 'model_performances.csv'  # Path to your CSV file
latex_output_file = 'table.csv'  # Output LaTeX file

# Convert the CSV to LaTeX
format_csv_column_names(csv_file_path, latex_output_file, decimal_places=5)

csv_file_path = 'f_score.csv'  # Path to your CSV file
latex_output_file = 'f_score.csv'  # Output LaTeX file
format_csv_column_names(csv_file_path, latex_output_file, decimal_places=5)