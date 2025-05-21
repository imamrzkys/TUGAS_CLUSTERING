import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def process_data(df):
    # Pilih fitur numerik yang digunakan untuk clustering
    features = ['Harga Satuan', 'Jumlah', 'Total', 'Modal', 'Laba Kotor', 'Rating']
    # Hapus data duplikat
    df = df.drop_duplicates()
    # Hapus data yang tidak lengkap (missing values)
    df = df.dropna()
    # Ambil data fitur numerik
    X = df[features].copy()
    # Normalisasi fitur numerik
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, features

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def elbow_method(X_scaled, max_k=10):
    inertias = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    fig1 = plt.figure()
    plt.plot(range(1, max_k+1), inertias, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method (Inertia vs K)')
    elbow_plot = plot_to_base64(fig1)
    fig2 = plt.figure()
    plt.plot(range(1, max_k+1), inertias, marker='o', color='orange')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method (WCSS vs K)')
    wcss_plot = plot_to_base64(fig2)
    return elbow_plot, wcss_plot

def run_kmeans(X_scaled, df, features, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df_result = df.copy()
    df_result['Cluster'] = labels
    fig = plt.figure()
    sns.scatterplot(x=df[features[0]], y=df[features[-1]], hue=labels, palette='Set1', s=60)
    plt.xlabel(features[0])
    plt.ylabel(features[-1])
    plt.title('K-Means Clustering Result (Unit price vs Rating)')
    scatter_plot = plot_to_base64(fig)
    return labels, scatter_plot, df_result

# === Pipeline utama untuk clustering offline dan simpan grafik ===
def train_and_plot_all(
    csv_path=r'c:/Users/X395/tugas/TUGAS6/product_clustering/dataset/preprocessed_data.csv',
    static_path=r'c:/Users/X395/tugas/TUGAS6/product_clustering/static',
    output_csv=r'c:/Users/X395/tugas/TUGAS6/product_clustering/dataset/clustered_data.csv',
    k_cluster=3,
    max_k=10
):
    print('Membaca data hasil preprocessing...')
    df = pd.read_csv(csv_path)
    features = ['Harga Satuan', 'Jumlah', 'Total', 'Modal', 'Laba Kotor', 'Rating']
    X = df[features].copy()
    X_scaled = X.values  # Sudah dinormalisasi di preprocessing.py

    # Elbow Method
    inertias = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    # Grafik 1: Elbow (Inertia vs K)
    fig1 = plt.figure()
    plt.plot(range(1, max_k+1), inertias, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method (Inertia vs K)')
    elbow_inertia_path = os.path.join(static_path, 'elbow_inertia_vs_k.png')
    fig1.savefig(elbow_inertia_path, bbox_inches='tight')
    plt.close(fig1)

    # Grafik 2: Elbow (WCSS vs K)
    fig2 = plt.figure()
    plt.plot(range(1, max_k+1), inertias, marker='o', color='orange')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method (WCSS vs K)')
    elbow_wcss_path = os.path.join(static_path, 'elbow_wcss_vs_k.png')
    fig2.savefig(elbow_wcss_path, bbox_inches='tight')
    plt.close(fig2)

    # Clustering dengan K=k_cluster
    kmeans = KMeans(n_clusters=k_cluster, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df['Cluster'] = labels

    # Grafik 3: Scatter plot hasil clustering (Unit price vs Rating)
    fig3 = plt.figure(figsize=(8,6))
    sns.scatterplot(x=df['Harga Satuan'], y=df['Rating'], hue=df['Cluster'], palette='Set1', s=60)
    plt.xlabel('Harga Satuan')
    plt.ylabel('Rating')
    plt.title('K-Means Clustering Result (Harga Satuan vs Rating)')
    plt.legend(title='Cluster')
    scatter_path = os.path.join(static_path, 'scatter_unitprice_rating_cluster.png')
    fig3.savefig(scatter_path, bbox_inches='tight')
    plt.close(fig3)

    # Simpan hasil clustering ke CSV
    df.to_csv(output_csv, index=False)
    print('Semua grafik dan hasil clustering telah disimpan.')

if __name__ == '__main__':
    train_and_plot_all()
