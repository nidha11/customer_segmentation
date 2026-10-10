import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# STEP 1: Create customer dataset

data = {
    "CustomerID": list(range(1, 21)),
    "Annual Income (k$)": [
        15, 16, 17, 18, 19,
        25, 28, 30, 35, 38,
        50, 55, 60, 65, 70,
        80, 85, 90, 95, 100
    ],
    "Spending Score (1-100)": [
        39, 81, 6, 77, 40,
        76, 60, 65, 35, 92,
        50, 55, 45, 70, 30,
        75, 20, 85, 15, 90
    ]
}

df = pd.DataFrame(data)

# Save dataset
df.to_csv("Mall_Customers.csv", index=False)

print("Dataset created successfully!")
print(df.head())


# STEP 2: Select features

X = df[
    ["Annual Income (k$)", "Spending Score (1-100)"]
]


# STEP 3: Scale data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# STEP 4: Apply K-Means clustering

print("Starting K-Means clustering...")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("Clustering completed!")


# STEP 5: Create graph

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
    s=100
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation Using K-Means")

plt.colorbar(label="Cluster")
plt.grid(True, alpha=0.3)
plt.tight_layout()


# STEP 6: Display graph

plt.show()


# STEP 7: Save clustering results

df.to_csv("Customer_Segmentation_Results.csv", index=False)

print("Graph displayed!")
print("Results saved successfully!")
print(df.head())