from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import pandas as pd
from model.clustering import process_data, elbow_method, run_kmeans

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
                           scatter_plot=scatter_plot,
                           table_html=table_html,
                           selected_k=selected_k,
                           categories=categories,
                           selected_category=selected_category or 'Semua')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
