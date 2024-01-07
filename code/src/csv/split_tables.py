import pandas as pd


def split_csv(file_path, n_splits):
    # Read the CSV file
    df = pd.read_csv(file_path, sep=';')

    # Common columns to be included in every split
    common_columns = df.iloc[:, 1:6]

    # Remaining columns to be split
    remaining_columns = df.iloc[:, 6:]

    # Calculate the number of columns in each split
    columns_per_split = remaining_columns.shape[1] // n_splits

    # Splitting the DataFrame
    split_dfs = []
    for i in range(n_splits):
        start_col = i * columns_per_split
        end_col = None if i == n_splits - 1 else start_col + columns_per_split
        split_cols = remaining_columns.iloc[:, start_col:end_col]
        split_df = pd.concat([common_columns, split_cols], axis=1)
        split_dfs.append(split_df)

    # Save each split to a separate CSV file
    for i, split_df in enumerate(split_dfs, start=1):
        split_df.to_csv(f'split_{i}.csv', sep=';', index=False)
    


file_path = 'transformed_data.csv'  # Update this path to your actual CSV file path
n_splits = 6  # Number of splits
split_csv(file_path, n_splits)