import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed dataset
df = pd.read_csv('preprocessed.csv')

# Step 1: Filter out cancelled orders (both 'c' and 'C')
df = df[~df['InvoiceNo'].str.startswith(('c', 'C'))]

# Step 2: Analyze Product Category Popularity
# Using the 'Description' column for product popularity
product_popularity = df['Description'].value_counts().reset_index()
product_popularity.columns = ['Product Description', 'Count']

# Plotting Product Popularity for Top 10 Products
plt.figure(figsize=(12, 8))
sns.barplot(x='Count', y='Product Description', data=product_popularity.head(10))  # Top 10 categories
plt.title('Top 10 Most Popular Product Categories')
plt.xlabel('Number of Purchases')
plt.ylabel('Product Description')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Adjust layout to fit labels better
plt.show()

# Print top 10 popular products with their counts
print("Top 10 Popular Products Sold:")
print(product_popularity.head(10))

# Print least 10 popular products with their counts
print("\nLeast 10 Popular Products Sold:")
print(product_popularity.tail(10))

# Step 3: Correlation Analysis between Frequency and Total Spending
# Create a DataFrame with Frequency and TotalSpend for correlation analysis
rfm_metrics = df.groupby('CustomerID').agg({
    'Frequency': 'first',          # Use existing Frequency column
    'TotalSpend': 'first'          # Use existing TotalSpend column
}).reset_index()

# Calculate correlation matrix
correlation_matrix = rfm_metrics.corr()

# Plotting Heatmap of Correlation Matrix with improved visibility
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap between Frequency and Total Spend')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.yticks(rotation=0)   # Keep y-axis labels horizontal
plt.tight_layout()       # Adjust layout to fit labels better
plt.show()

# Step 4: Analyze Products Sold by Quantity
# Aggregate total quantity sold for each product
quantity_sold = df.groupby('Description')['Quantity'].sum().reset_index()
quantity_sold.columns = ['Product Description', 'Total Quantity Sold']

# Sort by total quantity sold in descending order
quantity_sold = quantity_sold.sort_values(by='Total Quantity Sold', ascending=False)

# Print top 10 products with their total quantities sold
print("\nTop 10 Products Sold by Quantity:")
print(quantity_sold.head(10))

# Print least 10 products sold by quantity
print("\nLeast 10 Products Sold by Quantity:")
print(quantity_sold.tail(10))

# Plotting Total Quantity Sold for Top Products with improved y-axis label visibility
plt.figure(figsize=(12, 8))
sns.barplot(x='Total Quantity Sold', y='Product Description', data=quantity_sold.head(10))  # Top 10 products by quantity sold
plt.title('Top 10 Products Sold by Quantity')
plt.xlabel('Total Quantity Sold')
plt.ylabel('Product Description')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Adjust layout to fit labels better
plt.show()

# Calculate Total Spend by Country for Unique Customers
# Group by Country and CustomerID, then sum TotalSpend for each unique customer
unique_customer_spend = df.groupby(['Country', 'CustomerID'])['TotalSpend'].sum().reset_index()

# Now group by Country to get total spend from unique customers
country_sales = unique_customer_spend.groupby('Country')['TotalSpend'].sum().reset_index()
country_sales.columns = ['Country', 'Total Sales']

# Plotting Total Sales by Country
plt.figure(figsize=(12, 8))
sns.barplot(x='Total Sales', y='Country', data=country_sales.sort_values(by='Total Sales', ascending=False))
plt.title('Total Sales by Country (Based on Unique Customers)')
plt.xlabel('Total Sales')
plt.ylabel('Country')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 5: Customer Frequency by Country (Unique Customers)
country_frequency = df.groupby('Country')['CustomerID'].nunique().reset_index()
country_frequency.columns = ['Country', 'Unique Customers']

# Plotting Customer Frequency by Country
plt.figure(figsize=(12, 8))
sns.barplot(x='Unique Customers', y='Country', data=country_frequency.sort_values(by='Unique Customers', ascending=False))
plt.title('Unique Customers by Country')
plt.xlabel('Number of Unique Customers')
plt.ylabel('Country')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 6: Average Spend by Country (Based on Unique Customers)
country_average_spend = unique_customer_spend.groupby('Country')['TotalSpend'].mean().reset_index()
country_average_spend.columns = ['Country', 'Average Spend']

# Plotting Average Spend by Country
plt.figure(figsize=(12, 8))
sns.barplot(x='Average Spend', y='Country', data=country_average_spend.sort_values(by='Average Spend', ascending=False))
plt.title('Average Spend per Transaction by Country (Based on Unique Customers)')
plt.xlabel('Average Spend')
plt.ylabel('Country')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
