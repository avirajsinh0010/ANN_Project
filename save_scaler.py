import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


data = pd.read_csv("Mall_Customers.csv")


data = data.drop(['CustomerID'], axis=1)


data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})


scaler = StandardScaler()
features = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler.fit(features)

# Save the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("Scaler saved successfully!")
print(f"Scaler type: {type(scaler)}")