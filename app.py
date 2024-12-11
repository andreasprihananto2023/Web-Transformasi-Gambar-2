import streamlit as st
import cv2
import numpy as np

# Kompres gambar
@st.cache_data
def compress_image(image, max_size=(800, 800)):
    h, w = image.shape[:2]
    ratio = min(max_size[0]/w, max_size[1]/h)
    new_size = (int(w*ratio), int(h*ratio))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

# Caching
@st.cache_data
def transform_image(image, transform_type, **kwargs):
    if transform_type == 'translasi':
        dx, dy = kwargs.get('dx', 0), kwargs.get('dy', 0)
        matriks_translasi = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(image, matriks_translasi, (image.shape[1], image.shape[0]))
    
    elif transform_type == 'rotasi':
        sudut = kwargs.get('sudut', 0)
        tengah = (image.shape[1] // 2, image.shape[0] // 2)
        matriks_rotasi = cv2.getRotationMatrix2D(tengah, sudut, 1.0)
        return cv2.warpAffine(image, matriks_rotasi, (image.shape[1], image.shape[0]))
    
    elif transform_type == 'skala':
        skala_x, skala_y = kwargs.get('skala_x', 1.0), kwargs.get('skala_y', 1.0)
        return cv2.resize(image, None, fx=skala_x, fy=skala_y, interpolation=cv2.INTER_LINEAR)
    
    elif transform_type == 'distorsi':
        h, w = image.shape[:2]
        skew_x, skew_y = kwargs.get('skew_x', 0), kwargs.get('skew_y', 0)
        pts1 = np.float32([[0,0], [w-1,0], [0,h-1], [w-1,h-1]])
        pts2 = np.float32([[0,0], 
                           [w-1,0], 
                           [skew_x*w,h-1], 
                           [(1+skew_y)*w-1,h-1]])
        matriks_distorsi = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(image, matriks_distorsi, (w, h))

def main():
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman", ["Home Page", "Transformasi Gambar"])

    if page == "Landing Page":
        st.title("Selamat Datang di Website Transformasi Gambar")
        st.write("Sebuah website yang dibuat oleh Group 7: Andreas, Firdaus, dan Rizki")
        st.write("Sebuah Website yang memungkinkan Anda untuk mengunggah gambar dan menerapkan berbagai transformasi. Dibuat oleh Group 7: Andreas, Firdaus, dan Rizki")
        st.write("Klik tombol di bawah untuk mulai.")
        if st.button("Mulai Transformasi"):
            st.session_state.page = "Transformasi Gambar"
            st.experimental_rerun()

    elif page == "Transformasi Gambar":
        st.title("Transformasi Gambar Progress Group 7")

        # Unggah file
        unggah_file = st.file_uploader(
            "Unggah gambar dalam format JPEG atau PNG", 
            type=["jpg", "jpeg", "png"]
        )

        if unggah_file is not None:
            # Baca dan kompres gambar
            file_bytes = np.asarray(bytearray(unggah_file.read()), dtype=np.uint8)
            gambar_asli = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            gambar_asli = compress_image(gambar_asli)

            # Layout dengan gambar asli
            st.subheader("Gambar Asli")
            st.image(cv2.cvtColor(gambar_asli, cv2.COLOR_BGR2RGB), caption="Gambar Asli", use_container_width=True)

            # Pilih jenis transformasi
            transform_type = st.radio(
                "Pilih Jenis Transformasi",
                ['Translasi', 'Rotasi', 'Skala', 'Distorsi']
            )

            # Pengaturan transformasi
            st.subheader("Pilih Pengaturan Transformasi")
            if transform_type == 'Translasi':
                dx = st.slider("Translasi Horizontal (dx)", -200, 200, 50)
                dy = st.slider("Translasi Vertikal (dy)", -200, 200, 30)
                gambar_transformasi = transform_image(
                    gambar_asli, 
                    'translasi',
                                        dx=dx, 
                    dy=dy
                )

            elif transform_type == 'Rotasi':
                sudut = st.slider("Sudut Rotasi (derajat)", -180, 180, 45)
                gambar_transformasi = transform_image(
                    gambar_asli, 
                    'rotasi', 
                    sudut=sudut
                )

            elif transform_type == 'Skala':
                skala_x = st.slider("Skala Horizontal", 0.5, 3.0, 1.5)
                skala_y = st.slider("Skala Vertikal", 0.5, 3.0, 1.5)
                gambar_transformasi = transform_image(
                    gambar_asli, 
                    'skala', 
                    skala_x=skala_x, 
                    skala_y=skala_y
                )

            elif transform_type == 'Distorsi':
                skew_x = st.slider("Distorsi Horizontal", 0.0, 2.0, 1.5)
                skew_y = st.slider("Distorsi Vertikal", 0.0, 2.0, 0.5)
                gambar_transformasi = transform_image(
                    gambar_asli, 
                    'distorsi', 
                    skew_x=skew_x, 
                    skew_y=skew_y
                )

            # Gambar hasil transformasi
            st.subheader(f"Gambar Hasil Transformasi ({transform_type})")
            st.image(
                cv2.cvtColor(gambar_transformasi, cv2.COLOR_BGR2RGB), 
                caption=f"Gambar {transform_type}", 
                use_container_width=True)

if __name__ == "__main__":
    main()
