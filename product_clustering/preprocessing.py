import pandas as pd
import os
import random

# Path ke dataset asli dan hasil
csv_path = r'c:\Users\X395\tugas\TUGAS6\dataset\Supermarket Sales Cleaned.csv'
preprocessed_path = r'c:\Users\X395\tugas\TUGAS6\product_clustering\dataset\preprocessed_data.csv'

# Mapping kategori ke sub-produk detail (lebih banyak variasi)
detail_produk = {
    'Health and beauty': [
        'Lipstik', 'Bedak', 'Parfum', 'Masker Wajah', 'Skincare', 'Sabun Muka', 'Handbody', 'Serum', 'Toner', 'Shampoo',
        'Conditioner', 'Body Scrub', 'Sunscreen', 'Deodoran', 'Pelembab', 'Sabun Mandi', 'Pasta Gigi', 'Obat Jerawat', 'Hair Tonic', 'Hair Oil'
    ],
    'Electronic accessories': [
        'Headphone', 'Powerbank', 'Charger', 'Mouse', 'Keyboard', 'Speaker Bluetooth', 'Webcam', 'USB Hub', 'MicroSD', 'Earphone',
        'Wireless Charger', 'Smartwatch', 'Kabel Data', 'Adaptor', 'Lampu LED USB', 'Cooling Pad', 'VR Glasses', 'Bluetooth Receiver', 'Car Charger', 'Tripod Smartphone'
    ],
    'Home and lifestyle': [
        'Sprei', 'Bantal', 'Lemari', 'Kursi', 'Meja', 'Guling', 'Karpet', 'Dispenser', 'Vacuum Cleaner', 'Kipas Angin',
        'Blender', 'Setrika', 'Jam Dinding', 'Tirai', 'Rak Sepatu', 'Cermin', 'Alat Pel', 'Tempat Sampah', 'Kompor Gas', 'Rice Cooker'
    ],
    'Sports and travel': [
        'Sepatu Olahraga', 'Raket', 'Tas Hiking', 'Jersey', 'Matras Yoga', 'Bola Futsal', 'Sepeda', 'Helm Sepeda', 'Kacamata Renang', 'Jas Hujan',
        'Botol Minum', 'Compass', 'Sleeping Bag', 'Pedometer', 'Dumbbell', 'Skipping Rope', 'Pelindung Lutut', 'Pelampung', 'Topi Outdoor', 'Kaos Kaki Olahraga'
    ],
    'Food and beverages': [
        'Kopi', 'Teh', 'Roti', 'Jus', 'Susu', 'Coklat', 'Mie Instan', 'Biskuit', 'Sereal', 'Air Mineral',
        'Minuman Bersoda', 'Keripik Kentang', 'Permen', 'Susu Kental Manis', 'Madu', 'Selai Kacang', 'Susu UHT', 'Yogurt', 'Coklat Bubuk', 'Saus Tomat'
    ],
    'Fashion accessories': [
        'Jam Tangan', 'Kacamata', 'Dompet', 'Topi', 'Ikat Pinggang', 'Syal', 'Gelang', 'Cincin', 'Anting', 'Bros',
        'Tas Selempang', 'Tas Tangan', 'Pin', 'Kalung', 'Bandana', 'Jepit Rambut', 'Kaos Kaki', 'Sarung Tangan', 'Dasi', 'Clutch'
    ]
}

print('Membaca data...')
df = pd.read_csv(csv_path)
print('Menghapus data duplikat dan data kosong...')
df = df.drop_duplicates()
df = df.dropna()

# Ambil nama produk dan fitur numerik
product_lines = df['Product line'].reset_index(drop=True)
features = ['Unit price', 'Quantity', 'Total', 'cogs', 'gross income', 'Rating']
X = df[features].copy().reset_index(drop=True)

# Tambahkan kolom detail produk (Jenis Produk)
print('Menambahkan kolom detail produk...')
jenis_produk = [
    random.choice(detail_produk.get(line, ['Lainnya']))
    for line in product_lines
]

# Konversi harga ke IDR (misal dikali 15.000)
kurs = 15000
X['Unit price'] = (X['Unit price'] * kurs).round(0)
X['Total'] = (X['Total'] * kurs).round(0)
X['cogs'] = (X['cogs'] * kurs).round(0)
X['gross income'] = (X['gross income'] * kurs).round(0)

# Mapping rating ke 1–10
X['Rating'] = 1 + (X['Rating'] - X['Rating'].min()) * 9 / (X['Rating'].max() - X['Rating'].min())
X['Rating'] = X['Rating'].round(1)

# Gabungkan nama produk + jenis produk + fitur numerik asli (IDR) + rating 1-10
result = pd.concat([product_lines, pd.Series(jenis_produk, name='Jenis Produk'), X], axis=1)

# Ganti nama kolom ke Bahasa Indonesia
result.columns = [
    'Kategori Produk',
    'Jenis Produk',
    'Harga Satuan',
    'Jumlah',
    'Total',
    'Modal',
    'Laba Kotor',
    'Rating'
]

print('Menyimpan data hasil preprocessing...')
result.to_csv(preprocessed_path, index=False)
print(f'Preprocessing selesai! Data siap clustering: {preprocessed_path}')
print(f'Jumlah data: {result.shape[0]} baris')