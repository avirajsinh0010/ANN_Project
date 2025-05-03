import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Preprocess the data
# Drop irrelevant features
data = data.drop(['CustomerID'], axis=1)

# Encode gender
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Create scaler and fit it to the features
scaler = StandardScaler()
features = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler.fit(features)

# Save the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("Scaler saved successfully!")
print(f"Scaler type: {type(scaler)}")