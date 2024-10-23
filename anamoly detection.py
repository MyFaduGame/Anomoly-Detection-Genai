import google.generativeai as gemini
import numpy as np
from sklearn.cluster import KMeans

gemini.configure(api_key="Your-API-Key")

def get_embeddings(texts):
    """
    Function to get embeddings for a list of texts using the Google Generative AI API.
    """
    embeddings = []
    for text in texts:
        response = gemini.embed_content(
            model='models/text-embedding-004',
            content=text,
            task_type="clustering"
        )
        embeddings.append(response['embedding'])
    return np.array(embeddings)

def detect_anomalies(embeddings, n_clusters=3):
    """
    Function to detect anomalies using K-Means clustering.
    It assigns data points to a cluster and anomalies are data points that deviate from the cluster centers.
    """
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Compute the distances of each point from its assigned cluster center
    distances = np.linalg.norm(embeddings - kmeans.cluster_centers_[cluster_labels], axis=1)
    threshold = np.percentile(distances, 5)  # Set a threshold for anomaly detection (e.g., top 5% furthest points)
    
    # Anomalies are points whose distances exceed the threshold
    anomalies = distances > threshold
    return anomalies

if __name__ == "__main__":
    # Example data
    log_entries = [
        "User login successful from IP 192.168.1.1",
        "Failed login attempt from IP 10.0.0.5",
        "Server error: CPU overload at 100% its over heated",
        "Disk space running low on server",
        "User logout from IP 192.168.1.1",
        "everything is okay."
    ]
    
    # Step 1: Generate embeddings for log entries
    embeddings = get_embeddings(log_entries)

    # Step 2: Detect anomalies in the embeddings
    anomalies = detect_anomalies(embeddings)

    # Step 3: Output the result
    for i, is_anomaly in enumerate(anomalies):
        if is_anomaly:
            print(f"Anomaly detected in log entry: {log_entries[i]}")
        else:
            print(f"Log entry normal: {log_entries[i]}")
