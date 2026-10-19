import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load your dataset
file_path = 'Online Retail.xlsx'  # Specify your dataset file name here
df = pd.read_excel(file_path)

# Step 1: Data Cleaning
# Check for missing values
print("Missing values before cleaning:\n", df.isnull().sum())
#print(len(df))

# Create a mapping of InvoiceNo to valid CustomerID
invoice_customer_mapping = df[df['CustomerID'].notnull()].groupby('InvoiceNo')['CustomerID'].first()

# Fill missing CustomerIDs based on the mapping
df['CustomerID'] = df.apply(
    lambda row: invoice_customer_mapping[row['InvoiceNo']] if pd.isnull(row['CustomerID']) and row['InvoiceNo'] in invoice_customer_mapping.index else row['CustomerID'],
    axis=1
)

# Identify InvoiceNos with no valid CustomerIDs
invalid_invoice_nos = df[df['CustomerID'].isnull()]['InvoiceNo'].unique()

# Drop all records associated with those InvoiceNos
if len(invalid_invoice_nos) > 0:
    df = df[~df['InvoiceNo'].isin(invalid_invoice_nos)]

# Fill other missing values with 0 (if necessary)
df.fillna(0, inplace=True)

print("Missing values after cleaning:\n", df.isnull().sum())
#print(len(df))

# Step 2: Handle Negative Values
# Convert negative values in Quantity and UnitPrice to absolute values
df['Quantity'] = df['Quantity'].abs()
df['UnitPrice'] = df['UnitPrice'].abs()

# Step 3: Create TotalSpend Column
df['TotalSpend'] = df['Quantity'] * df['UnitPrice']

# Step 4: Address Outliers (if necessary)
# Use IQR method to remove outliers in 'TotalSpend' column
Q1 = df['TotalSpend'].quantile(0.25)
Q3 = df['TotalSpend'].quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers
df = df[(df['TotalSpend'] >= (Q1 - 1.5 * IQR)) & (df['TotalSpend'] <= (Q3 + 1.5 * IQR))]

# Step 5: Feature Creation for RFM Analysis
# Convert 'InvoiceDate' to datetime format if not already done
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Define current date as one day after the most recent InvoiceDate
most_recent_date = df['InvoiceDate'].max()
current_date = most_recent_date + pd.Timedelta(days=1)
print(current_date)

# Create RFM metrics and add them as new columns to the original DataFrame
rfm_metrics = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (current_date - x.max()).days,  # Recency
    'InvoiceNo': 'count',  # Frequency
    'TotalSpend': 'sum'  # Monetary Value
}).reset_index()

# Rename columns for clarity in the new DataFrame
rfm_metrics.columns = ['CustomerID', 'Recency', 'Frequency', 'MonetaryValue']

# Merge RFM metrics back into the original DataFrame based on CustomerID
df = pd.merge(df, rfm_metrics, on='CustomerID', how='left')

# Debugging outputs to check RFM values before normalization
print("DataFrame with RFM metrics added:\n", df.head())

# Step 6: Z-score Normalization of RFM Metrics
scaler = StandardScaler()

# Select RFM columns for normalization
rfm_columns = ['Recency', 'Frequency', 'MonetaryValue']

# Apply Z-score normalization and add normalized values to the DataFrame
normalized_rfm_values = scaler.fit_transform(df[rfm_columns])
df[['Normalized_Recency', 'Normalized_Frequency', 'Normalized_MonetaryValue']] = normalized_rfm_values

# Output the normalized DataFrame for verification
print("DataFrame after Z-score normalization:\n", df[['Normalized_Recency', 'Normalized_Frequency', 'Normalized_MonetaryValue']].head())

# Save the preprocessed DataFrame to a CSV file
output_file_path = 'preprocessed.csv'
df.to_csv(output_file_path, index=False)
print(f"Preprocessed data saved to {output_file_path}.")
