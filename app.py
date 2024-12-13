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
    # Navigation bar di atas
    if 'page' not in st.session_state:
        st.session_state.page = "Home Page"
    
    # Membuat layout untuk navigation bar
    nav_bar = st.container()
    col1, col2 = nav_bar.columns([1, 3])  # Kolom untuk logo dan tombol

    with col1:
        st.write("")

    with col2:
        # Membuat dua kolom untuk tombol
        button_col1, button_col2 = st.columns(2)
        with button_col1:
            if st.button("Home Page"):
                st.session_state.page = "Home Page"
                st.experimental_rerun()
        
        with button_col2:
            if st.button("Transformasi Gambar"):
                st.session_state.page = "Transformasi Gambar"
                st.experimental_rerun()

    # Debugging: Tampilkan nilai dari st.session_state.page
    st.write("Current Page:", st.session_state.page)
    if st.session_state.page == "Home Page":
        # Membuat dua kolom
        col1, col2 = st.columns([0.6, 2])  # Kolom 1 lebih kecil dari kolom 2

        with col1:
            # Menampilkan gambar di kolom pertama
            st.image("logo_pu.png", caption="President University", width=110)
        with col2:
            # Menampilkan teks di kolom kedua
            st.markdown("<h1 style='font-size: 40px;'>PRESIDENT UNIVERSITY</h1>", unsafe_allow_html=True)
            st.markdown("<h2 style='font-size: 30px;'>Teknik Industri - Fakultas Teknik</h2>", unsafe_allow_html=True)
        
        st.title("Selamat Datang di Website Transformasi Gambar Group 7")    
        st.write("Website ini memungkinkan untuk mengunggah gambar dan menerapkan berbagai transformasi Geometrik. Dibuat oleh Andreas, Firdaus, Rizki")
        st.write(" ")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.image("foto_rizki.jpg", caption="Ahmad Rizki Safei", use_container_width=True)
        with col2:
            st.image("foto_andre.jpg", caption="Andreas Prihananto", use_container_width=True)
        with col3:
            st.image("foto_firdaus.jpg", caption="Firdaus Bachtiar", use_container_width=True)

        st.write("")
        st.write("Klik tombol di bawah untuk mulai.")
        if st.button("Mulai Transformasi"):
            st.session_state.page = "Transformasi Gambar"
            st.experimental_rerun()

    elif st.session_state.page == "Transformasi Gambar":
        # Tambahkan tombol kembali ke halaman utama
        if st.button("Kembali ke Halaman Utama"):
            st.session_state.page = "Home Page"
            st.experimental_rerun()

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

            # Pilih jenis transformasi dengan dropdown
            transform_type = st.selectbox("Pilih Jenis Transformasi", ["translasi", "rotasi", "skala", "distorsi", "saturasi"])

            # Real-time transformasi dengan slider
            if transform_type == "translasi":
                # Slider untuk translasi
                dx = st.slider("Translasi X (dx)", min_value=-200, max_value=200, value=0, step=10)
                dy = st.slider("Translasi Y (dy)", min_value=-200, max_value=200, value=0, step=10)
                
                # Transformasi real-time
                gambar_transformed = transform_image(gambar_asli, transform_type, dx=dx, dy=dy)
                
                with col2:
                    st.image(cv2.cvtColor(gambar_transformed, cv2.COLOR_BGR2RGB), 
                             caption=f"Translasi (dx={dx}, dy={dy})", 
                             use_container_width=True)

            elif transform_type == "rotasi":
                # Slider untuk rotasi
                sudut = st.slider("Sudut Rotasi (derajat)", min_value=-180, max_value=180, value=0, step=10)
                
                # Transformasi real-time
                gambar_transformed = transform_image(gambar_asli, transform_type, sudut=sudut)
                
                with col2:
                    st.image(cv2.cvtColor(gambar_transformed, cv2.COLOR_BGR2RGB), 
                             caption=f"Rotasi (sudut={sudut}°)", 
                             use_container_width=True)

            elif transform_type == "skala":
                # Slider untuk skala
                skala_x = st.slider("Skala X", min_value=0.1, max_value=2.0, value=1.)
                skala_y = st.slider("Skala Y", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
                
                # Transformasi real-time
                gambar_transformed = transform_image(gambar_asli, transform_type, skala_x=skala_x, skala_y=skala_y)
                
                with col2:
                    st.image(cv2.cvtColor(gambar_transformed, cv2.COLOR_BGR2RGB), 
                             caption=f"Skala (x={skala_x}, y={skala_y})", 
                             use_container_width=True)

            elif transform_type == "distorsi":
                # Slider untuk distorsi
                skew_x = st.slider("Distorsi X", min_value=-0.5, max_value=0.5, value=0.0, step=0.1)
                skew_y = st.slider("Distorsi Y", min_value=-0.5, max_value=0.5, value=0.0, step=0.1)
                
                # Transformasi real-time
                gambar_transformed = transform_image(gambar_asli, transform_type, skew_x=skew_x, skew_y=skew_y)  
                
                with col2:
                    st.image(cv2.cvtColor(gambar_transformed, cv2.COLOR_BGR2RGB), 
                            caption=f"Distorsi (x={skew_x}, y={skew_y})", 
                            use_container_width=True)

            elif transform_type == "saturasi":
                # Slider untuk saturasi
                saturasi = st.slider("Saturasi", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
                # Konversi gambar ke HSV
                hsv = cv2.cvtColor(gambar_asli, cv2.COLOR_BGR2HSV)
                # Atur saturasi
                hsv[..., 1] = np.clip(hsv[..., 1] * saturasi, 0, 255)
                # Konversi kembali ke BGR
                gambar_transformed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                with col2:
                    st.image(cv2.cvtColor(gambar_transformed, cv2.COLOR_BGR2RGB), 
                     caption=f"Saturasi (saturasi={saturasi})", 
                     use_container_width=True)

            if 'gambar_transformed' in locals():
                # Simpan gambar yang ditransformasikan ke dalam buffer
                _, buffer = cv2.imencode('.png', gambar_transformed)
                img_bytes = buffer.tobytes()

                # Tombol untuk mengunduh gambar
                st.download_button(
                    label="Unduh Hasil",
                    data=img_bytes,
                    file_name="hasil_gambar.png",
                    mime="image/png")

if __name__ == "__main__":
    main()
