import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from PIL.ExifTags import TAGS
from sklearn.cluster import KMeans

st.set_page_config(initial_sidebar_state="collapsed")

# Kompres gambar
@st.cache_data
def compress_image(image, max_size=(800, 800)):
    h, w = image.shape[:2]
    ratio = min(max_size[0]/w, max_size[1]/h)
    new_size = (int(w*ratio), int(h*ratio))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

# Caching
@st.cache_data
def transform_image(image, dx=0, dy=0, sudut=0, skala_x=1.0, skala_y=1.0, skew_x=0, skew_y=0, blur_kernel=1, saturation=1.0):
    # Translasi
    matriks_translasi = np.float32([[1, 0, dx], [0, 1, dy]])
    image = cv2.warpAffine(image, matriks_translasi, (image.shape[1], image.shape[0]))

    # Rotasi
    tengah = (image.shape[1] // 2, image.shape[0] // 2)
    matriks_rotasi = cv2.getRotationMatrix2D(tengah, sudut, 1.0)
    image = cv2.warpAffine(image, matriks_rotasi, (image.shape[1], image.shape[0]))

    # Skala
    image = cv2.resize(image, None, fx=skala_x, fy=skala_y, interpolation=cv2.INTER_LINEAR)

    # Distorsi
    h, w = image.shape[:2]
    pts1 = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    pts2 = np.float32([[0, 0], 
                        [w-1, 0], 
                        [skew_x * w, h-1], 
                        [(1 + skew_y) * w - 1, h-1]])
    matriks_distorsi = cv2.getPerspectiveTransform(pts1, pts2)
    image = cv2.warpPerspective(image, matriks_distorsi, (w, h))

    # Gaussian Blur
    if blur_kernel > 1:
        image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)

    # Saturasi
    if saturation != 1.0:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[..., 1] = np.clip(hsv_image[..., 1] * saturation, 0, 255)  # Mengatur saturasi
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return image

def edge_detection(image, threshold1=100, threshold2=200):
    return cv2.Canny(image, threshold1, threshold2)

def extract_metadata(image):
    metadata = {}
    if hasattr(image, '_getexif'):
        exif_data = image._getexif()
        if exif_data is not None:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                metadata[tag] = value
        else:
            # Ganti pesan ini
            metadata['Info'] = 'Gambar pernah diedit sebelumnya, tidak ada metadata yang dapat ditampilkan'
    else:
        metadata['Info'] = 'Gambar pernah diedit sebelumnya, tidak ada metadata yang dapat ditampilkan'
    return metadata

def get_dominant_color(image, k=5):
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Fit the KMeans model
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Get the most common color
    dominant_color = kmeans.cluster_centers_[kmeans.labels_].astype(int)
    
    # Count the occurrences of each color
    unique_colors, counts = np.unique(dominant_color, axis=0, return_counts=True)
    
    # Get the index of the most common color
    dominant_color_index = counts.argmax()
    
    return unique_colors[dominant_color_index]

def remove_non_dominant_colors(image, dominant_color, threshold=60):
    # Buat batas bawah dan atas untuk warna dominan
    lower_bound = np.array([max(0, c - threshold) for c in dominant_color])
    upper_bound = np.array([min(255, c + threshold) for c in dominant_color])
    
    # Buat mask untuk warna yang dekat dengan warna dominan
    mask = cv2.inRange(image, lower_bound, upper_bound)
    
    # Buat gambar output dengan warna non-dominan diatur ke putih
    output_image = np.full(image.shape, 255, dtype=np.uint8)  # Mulai dengan gambar putih
    output_image[mask > 0] = image[mask > 0]  # Set warna dominan

    return output_image

def main():
    st.sidebar.title("Group 7")
    if 'page' not in st.session_state:
        st.session_state.page = "Home Page"

    page = st.sidebar.radio("Pilih Halaman", ["Home Page", "Transformasi Geometrik", "Ekstraksi Gambar"], index=["Home Page", "Transformasi Geometrik", "Ekstraksi Gambar"].index(st.session_state.page))
    
    # Update session state with the selected page
    st.session_state.page = page

    if st.session_state.page == "Home Page":
        # Membuat dua kolom
        col1, col2, col3 = st.columns([0.6, 0.1, 2])  # Kolom 1 lebih kecil dari kolom 2

        with col1:
            # Menampilkan gambar di kolom pertama
            st.image("logo_pu.png", caption="President University", width=120)
        with col2:
            st.write("")
        with col3:
            # Menampilkan teks di kolom kedua
            st.markdown("<h1 style='font-size: 40px;'>PRESIDENT UNIVERSITY</h1>", unsafe_allow_html=True)
            st.markdown("<h2 style='font-size: 28px;'>Aljabar Linear - Teknik Industri</h2>", unsafe_allow_html=True)
        
        st.title("Selamat Datang di Website Transformasi Gambar Group 7")    
        st.write("Website ini memungkinkan untuk mengunggah gambar dan menerapkan berbagai transformasi Gambar menggunakan Python. Dibuat oleh Andreas, Firdaus, Rizki")
        st.write(" ")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.image("foto rizki.jpg", caption="Ahmad Rizki Safei", use_container_width=True)
        with col2:
            st.image("foto andre.jpg", caption="Andreas Prihananto", use_container_width=True)
        with col3:
            st.image("foto firdaus.jpg", caption="Firdaus Bachtiar", use_container_width=True)

        st.write("")
        st.write("Klik tombol di bawah untuk mulai.")

        col1, col2 = st.columns([0.8, 1])
        with col1:
            if st.button("Transformasi Geometrik"):
                st.session_state.page = "Transformasi Geometrik"
                st.rerun()
        with col2:
            if st.button("Ekstrasi Gambar"):
                st.session_state.page = "Ekstraksi Gambar"
                st.rerun()

    elif st.session_state.page == "Transformasi Geometrik":
        # Tambahkan tombol kembali ke halaman utama
        if st.button("Kembali ke Halaman Utama"):
            st.session_state.page = "Home Page"
            st.rerun()

        st.title("Transformasi Gambar")
        uploaded_file = st.file_uploader("Unggah Gambar yang akan ditransformasilan", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Decode gambar
            gambar_asli = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            gambar_asli = compress_image(gambar_asli)

            # Kolom untuk menampilkan gambar asli dan hasil transformasi
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(cv2.cvtColor(gambar_asli, cv2.COLOR_BGR2RGB), caption="Gambar Asli", use_container_width=True)

            # Slider untuk translasi
            dx = st.slider("Translasi X (dx)", min_value=-200, max_value=200, value=0, step=10)
            dy = st.slider("Translasi Y (dy)", min_value=-200, max_value=200, value=0, step=10)

            # Slider untuk rotasi
            sudut = st.slider("Sudut Rotasi (derajat)", min_value=-180, max_value=180, value=0, step=10)

            # Slider untuk skala
            skala_x = st.slider("Skala X", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
            skala_y = st.slider("Skala Y", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

            # Slider untuk Skewing
            skew_x = st.slider("Skewing X", min_value=-0.5, max_value=0.5, value=0.0, step=0.1)
            skew_y = st.slider("Skewing Y", min_value=-0.5, max_value=0.5, value=0.0, step=0.1)

            # Slider untuk Gaussian Blur
            blur_kernel = st.slider("Gaussian Blur", min_value=1, max_value=21, value=1, step=2)

            # Slider untuk saturasi
            saturation = st.slider("Saturasi", min_value=0.0, max_value=3.0, value=1.0, step=0.1)

            # Terapkan semua transformasi
            gambar_transformed = transform_image(gambar_asli, dx=dx, dy=dy, sudut=sudut, skala_x=skala_x, skala_y=skala_y, skew_x=skew_x, skew_y=skew_y, blur_kernel=blur_kernel, saturation=saturation)

            with col2:
                st.image(cv2.cvtColor(gambar_transformed, cv2.COLOR_BGR2RGB), caption="Gambar Ditransformasi", use_container_width=True)

            if gambar_transformed is not None:
                # Simpan gambar yang ditransformasikan ke dalam buffer
                _, buffer = cv2.imencode('.png', gambar_transformed)
                img_bytes = buffer.tobytes()

                # Tombol untuk mengunduh gambar
                st.download_button(
                    label="Unduh Hasil",
                    data=img_bytes,
                    file_name="hasil_gambar.png",
                    mime="image/png")

    elif st.session_state.page == "Ekstraksi Gambar":
        # Tambahkan tombol kembali ke halaman utama
        if st.button("Kembali ke Halaman Utama"):
            st.session_state.page = "Home Page"
            st.rerun()

        st.title("Ekstraksi Gambar")
        uploaded_file = st.file_uploader("Unggah Gambar untuk Ekstraksi Gambar", type=["jpg", "jpeg", "png"])
        gambar_ekstraksi_gambar = None
        if uploaded_file is not None:
            # Decode gambar
            gambar_asli = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            gambar_asli = compress_image(gambar_asli)

            # Menggunakan PIL untuk membaca metadata
            pil_image = Image.open(uploaded_file)
            metadata = extract_metadata(pil_image)

            # Dapatkan warna dominan
            dominant_color = get_dominant_color(gambar_asli)

            # Kolom untuk menampilkan gambar asli dan hasil ekstraksi tepi
            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(cv2.cvtColor(gambar_asli, cv2.COLOR_BGR2RGB), caption="Gambar Asli", use_container_width=True)

            st.subheader("Atur Sensitivitas Ekstraksi")
            # Slider untuk threshold Canny
            threshold1 = st.slider("Threshold 1", min_value=0, max_value=255, value=100, step=1)
            threshold2 = st.slider("Threshold 2", min_value=0, max_value=255, value=200, step=1)

            # Ekstraksi tepi
            gambar_ekstraksi_gambar = edge_detection(gambar_asli, threshold1=threshold1, threshold2=threshold2)

            with col2:
                st.image(gambar_ekstraksi_gambar, caption="Hasil Ekstraksi Gambar", use_container_width=True)

            with col3:
            # Hapus warna non-dominan
                gambar_dominan = remove_non_dominant_colors(gambar_asli, dominant_color)
            # Tampilkan gambar dengan warna dominan saja
                if gambar_dominan is not None and gambar_dominan.size > 0:
                    st.image(cv2.cvtColor(gambar_dominan, cv2.COLOR_BGR2RGB), caption="Sebaran Warna Dominan", use_container_width=True)
                else:
                    st.error("Gambar dengan warna dominan tidak dapat ditampilkan.")
            
            # Tampilkan metadata
            st.subheader("Metadata Gambar")
            st.write("Khusus gambar asli, tanpa edit sebelumnya")
            for key, value in metadata.items():
                st.write(f"{key}: {value}")

    
            # Hapus warna non-dominan
            gambar_dominan = remove_non_dominant_colors(gambar_asli, dominant_color)
            st.write("")        
            
        if gambar_ekstraksi_gambar is not None:
            # Simpan gambar hasil ekstraksi tepi ke dalam buffer
            _, buffer = cv2.imencode('.png', gambar_ekstraksi_gambar)
            img_bytes = buffer.tobytes()

            # Tombol untuk mengunduh gambar hasil ekstraksi tepi
            st.download_button(
                label="Unduh Hasil Ekstraksi Gambar",
                data=img_bytes,
                file_name="hasil_ekstraksi_gambar.png",
                mime="image/png")

            # Simpan gambar dominan ke dalam buffer
            _, buffer_dominan = cv2.imencode('.png', gambar_dominan)
            img_bytes_dominan = buffer_dominan.tobytes()
    
            # Tombol untuk mengunduh gambar dominan
            st.download_button(
                label="Unduh Sebaran Warna Dominan",
                data=img_bytes_dominan,
                file_name="Warna_dominan.png",
                mime="image/png")

if __name__ == "__main__":
    main()
