# MusicRec - Spotify-Inspired Music Recommender UI

Aplikasi web berbasis Streamlit yang menampilkan UI rekomendasi musik dengan inspirasi dari Spotify. Aplikasi ini mendemonstrasikan dua jenis sistem rekomendasi: Collaborative Filtering dan Content-Based Filtering.

## Fitur

- 🔐 **Autentikasi Sederhana**: Login dan registrasi user
- 🎨 **Dual Theme**: Mode dark dan light yang dapat di-toggle
- 🎵 **Rekomendasi Music**: Tampilan horizontal scrollable untuk rekomendasi
- 📊 **Dua Algoritma**: Tab terpisah untuk Collaborative dan Content-Based recommendations
- 🎯 **UI Modern**: Desain terinspirasi dari Spotify dengan navigasi hybrid

## Struktur Proyek

```
app/
├── app.py                          # Entry point (login/register)
├── pages/
│   └── 1_home.py                  # Halaman rekomendasi utama
├── components/
│   ├── navbar.py                  # Top navigation bar
│   ├── sidebar.py                 # Sidebar navigation
│   ├── recommendation_row.py      # Komponen row rekomendasi
│   └── theme_toggle.py            # Toggle tema
├── utils/
│   ├── auth.py                    # Helper autentikasi
│   ├── mock_data.py               # Data mock untuk demo
│   └── session.py                 # Session state management
└── styles/
    ├── dark_theme.css             # CSS tema dark
    └── light_theme.css            # CSS tema light
```

## Instalasi

1. Clone repository ini
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Cara Menjalankan

Jalankan aplikasi dengan perintah:

```bash
cd app
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## Cara Menggunakan

1. **Register**: Buat akun baru dengan username dan password
2. **Login**: Login dengan kredensial yang sudah dibuat
3. **Explore**: Lihat rekomendasi musik di dua tab berbeda:
   - **Collaborative Filtering**: Rekomendasi berdasarkan user dengan selera serupa
   - **Content-Based**: Rekomendasi berdasarkan preferensi musik Anda
4. **Toggle Theme**: Klik tombol ☀️/🌙 untuk mengganti tema
5. **Logout**: Klik tombol Logout untuk keluar

## Catatan

- Ini adalah **skeleton UI** untuk demonstrasi - tidak ada model ML yang sebenarnya
- Data rekomendasi menggunakan **mock data** yang hardcoded
- Autentikasi disimpan di **session state** (hilang saat refresh)
- Gambar album menggunakan **placeholder** dari via.placeholder.com

## Teknologi

- **Streamlit**: Framework web app
- **Python 3.7+**: Bahasa pemrograman
- **CSS**: Custom styling untuk tema Spotify-inspired

## Pengembangan Selanjutnya

Untuk mengembangkan aplikasi ini lebih lanjut:

1. Integrasikan dengan model ML untuk rekomendasi real
2. Tambahkan database untuk menyimpan user dan preferensi
3. Implementasikan fitur play music
4. Tambahkan fitur playlist dan library
5. Integrasikan dengan API musik (Spotify, Last.fm, dll)

## Lisensi

Project ini dibuat untuk keperluan akademis.
