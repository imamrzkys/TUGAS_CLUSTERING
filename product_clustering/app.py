from flask import Flask, render_template, request, redirect, url_for, flash
import os
from sklearn.cluster import KMeans
from werkzeug.utils import secure_filename
import pandas as pd
from model.clustering import process_data, elbow_method, run_kmeans, plot_to_base64
import matplotlib.pyplot as plt

DATA_PATH = 'product_clustering/dataset/preprocessed_data.csv'

app = Flask(__name__)
app.secret_key = 'supersecretkey'

@app.route('/', methods=['GET', 'POST'])
def index():
    elbow_plot = None
    wcss_plot = None
    scatter_plot = None
    table_html = None
    selected_k = 3  # default
    selected_category = request.form.get('category') if request.method == 'POST' else None
    df = pd.read_csv(DATA_PATH)
    categories = sorted(df['Product line'].unique()) if 'Product line' in df.columns else []
    # Filter kategori jika dipilih
    if selected_category and selected_category != 'Semua':
        df = df[df['Product line'] == selected_category]
    X_scaled, features = process_data(df)
    elbow_plot, wcss_plot = elbow_method(X_scaled)

    # Silhouette Score plot (base64)
    from sklearn.metrics import silhouette_score
    silhouette_scores = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
    fig_sil = plt.figure()
    plt.plot(list(k_range), silhouette_scores, marker='o')
    plt.xlabel('Jumlah Klaster (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Metode Silhouette Score')
    silhouette_plot = plot_to_base64(fig_sil)

    if request.method == 'POST' and 'k_value' in request.form:
        selected_k = int(request.form['k_value'])
        cluster_labels, scatter_plot, df_result = run_kmeans(X_scaled, df, features, selected_k)
        # Filter tabel juga agar sesuai filter kategori
        if selected_category and selected_category != 'Semua':
            df_result = df_result[df_result['Product line'] == selected_category]
        table_html = df_result.to_html(classes='table table-striped table-hover align-middle aesthetic-table', index=False, border=0)
    else:
        cluster_labels, scatter_plot, df_result = run_kmeans(X_scaled, df, features, selected_k)
        table_html = df_result.to_html(classes='table table-striped table-hover align-middle aesthetic-table', index=False, border=0)
    return render_template('index.html',
                           elbow_plot=elbow_plot,
                           wcss_plot=wcss_plot,
                           silhouette_plot=silhouette_plot,
                           scatter_plot=scatter_plot,
                           table_html=table_html,
                           selected_k=selected_k,
                           categories=categories,
                           selected_category=selected_category or 'Semua')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

from flask import send_file
import io

@app.route('/download/<plot_type>')
def download_plot(plot_type):
    X_scaled, features = process_data(pd.read_csv('product_clustering/dataset/preprocessed_data.csv'))
    buf = io.BytesIO()
    if plot_type == 'elbow':
        elbow_plot, _ = elbow_method(X_scaled)
        # regenerate plot as PNG
        inertias = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        fig = plt.figure()
        plt.plot(range(1, 11), inertias, marker='o')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method (Inertia vs K)')
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
    elif plot_type == 'wcss':
        _, wcss_plot = elbow_method(X_scaled)
        inertias = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        fig = plt.figure()
        plt.plot(range(1, 11), inertias, marker='o', color='orange')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('WCSS')
        plt.title('Elbow Method (WCSS vs K)')
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
    elif plot_type == 'silhouette':
        from sklearn.metrics import silhouette_score
        silhouette_scores = []
        k_range = range(2, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
        fig = plt.figure()
        plt.plot(list(k_range), silhouette_scores, marker='o')
        plt.xlabel('Jumlah Klaster (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Metode Silhouette Score')
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
    elif plot_type == 'scatter':
        df = pd.read_csv('product_clustering/dataset/preprocessed_data.csv')
        X_scaled, features = process_data(df)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        df['Cluster'] = labels
        fig = plt.figure(figsize=(8,6))
        sns.scatterplot(x=df['Harga Satuan'], y=df['Rating'], hue=df['Cluster'], palette='Set1', s=60)
        plt.xlabel('Harga Satuan')
        plt.ylabel('Rating')
        plt.title('K-Means Clustering Result (Harga Satuan vs Rating)')
        plt.legend(title='Cluster')
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
    else:
        return 'Plot type not found', 404
    buf.seek(0)
    return send_file(buf, mimetype='image/png', as_attachment=True, download_name=f'{plot_type}_plot.png')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
