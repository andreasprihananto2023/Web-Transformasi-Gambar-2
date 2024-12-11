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
    if 'page' not in st.session_state:
        st.session_state.page = "Landing Page"

    page = st.sidebar.radio("Pilih Halaman", ["Landing Page", "Transformasi Gambar"], index=["Landing Page", "Transformasi Gambar"].index(st.session_state.page))
    
    # Update session state with the selected page
    st.session_state.page = page

    if st.session_state.page == "Landing Page":
        st.title("Selamat Datang di Aplikasi Transformasi Gambar")
        st.write("Aplikasi ini memungkinkan Anda untuk mengunggah gambar dan menerapkan berbagai transformasi.")
        st.write("Klik tombol di bawah untuk mulai.")
        if st.button("Mulai Transformasi"):
            st.session_state.page = "Transformasi Gambar"
            st.experimental_rerun()

    elif st.session_state.page == "Transformasi Gambar":
        st.title("Transformasi Gambar")
        uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            gambar_asli = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            gambar_asli = compress_image(gambar_asli)

            st.image(cv2.cvtColor(gambar_asli, cv2.COLOR_BGR2RGB), caption="Gambar Asli", use_container_width=True)

            transform_type = st.selectbox("Pilih Jenis Transformasi", ["translasi", "rotasi", "skala", "distorsi"])

            if transform_type == "translasi":
                dx = st.number_input("Masukkan nilai translasi X (dx)", value=0)
                dy = st.number_input("Masukkan nilai translasi Y (dy)", value=0)
                if st.button("Terapkan Transformasi"):
                    gambar_transformed = transform_image(gambar_asli, transform_type, dx=dx, dy=dy)
                    st.image(cv2.cvtColor(gambar_transformed, cv2.COLOR_BGR2RGB), caption="Gambar Setelah Translasi", use_container_width=True)

            elif transform_type == "rotasi":
                sudut = st.number_input("Masukkan sudut rotasi (dalam derajat)", value=0)
                if st.button("Terapkan Transformasi"):
                    gambar_transformed = transform_image(gambar_asli, transform_type, sudut=sudut)
                    st.image(cv2.cvtColor(gambar_transformed, cv2.COLOR_BGR2RGB), caption="Gambar Setelah Rotasi", use_container_width=True)

            elif transform_type == "skala":
                skala_x = st.number_input("Masukkan skala X", value=1.0)
                skala_y = st.number_input("Masukkan skala Y", value=1.0)
                if st.button("Terapkan Transformasi"):
                    gambar_transformed = transform_image(gambar_asli, transform_type, skala_x=skala_x, skala_y=skala_y)
                    st.image(cv2.cvtColor(gambar_transformed, cv2.COLOR_BGR2RGB), caption="Gambar Setelah Skala", use_container_width=True)

            elif transform_type == "distorsi":
                skew_x = st.number_input("Masukkan nilai distorsi X", value=0.0)
                skew_y = st.number_input("Masukkan nilai distorsi Y", value=0.0)
                if st.button("Terapkan Transformasi"):
                    gambar_transformed = transform_image(gambar_asli, transform_type, skew_x=skew_x, skew_y=skew_y)
                    st.image(cv2.cvtColor(gambar_transformed, cv2.COLOR_BGR2RGB), caption="Gambar Setelah Distorsi", use_container_width=True)

if __name__ == "__main__":
    main()
