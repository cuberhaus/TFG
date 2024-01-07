# SEGON
import pandas as pd

# Let's assume 'data.csv' is the name of the CSV file you have, which contains the data.
# We will read the CSV, replace '.' with ',' in the numeric values, and then export it with ';' as the separator.

# Read the CSV file
# df = pd.read_csv('./model_performances.csv')
df = pd.read_csv('./table.csv')

# Replace '.' with ',' in the numeric columns
df = df.applymap(lambda x: str(x).replace('.', ',') if isinstance(x, (float, int)) else x)

# Save the transformed data to a new CSV file with ';' as the separator
output_csv_path = './transformed_data.csv'
df.to_csv(output_csv_path, sep=';', index=False)

output_csv_path


df = pd.read_csv('./f_score.csv')

# Replace '.' with ',' in the numeric columns
df = df.applymap(lambda x: str(x).replace('.', ',') if isinstance(x, (float, int)) else x)

# Save the transformed data to a new CSV file with ';' as the separator
output_csv_path = './f_score.csv'
df.to_csv(output_csv_path, sep=';', index=False)