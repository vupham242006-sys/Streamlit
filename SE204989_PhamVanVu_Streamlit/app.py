# -*- coding: utf-8 -*-
"""
Demo Phân Loại Hoa Iris bằng Streamlit (Gaussian Naive Bayes)
Sinh viên: Phạm Văn Vũ - MSSV: SE204989
Chạy ứng dụng:  streamlit run app.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# ============================================================
# 1. CẤU HÌNH TRANG (tương đương gr.Blocks(title=..., theme=...))
# ============================================================
st.set_page_config(page_title="Iris Classification - Naive Bayes", page_icon="🌸", layout="wide")

# ============================================================
# 2. HẰNG SỐ: thông số slider, đường dẫn ảnh
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "hoa_Iris")  # thư mục chứa ảnh minh họa

# Mỗi slider: (nhãn tiếng Việt kèm tiếng Anh, min, max, mặc định, bước)
SLIDERS = [
    ("Chiều dài lá đài (Sepal Length - cm)", 4.0, 8.0, 5.1, 0.1),
    ("Chiều rộng lá đài (Sepal Width - cm)", 2.0, 4.5, 3.5, 0.1),
    ("Chiều dài cánh hoa (Petal Length - cm)", 1.0, 7.0, 1.4, 0.1),
    ("Chiều rộng cánh hoa (Petal Width - cm)", 0.1, 2.5, 0.2, 0.1),
]

SPECIES_IMAGES = {
    "setosa": os.path.join(IMG_DIR, "Iris setosa.jpg"),
    "versicolor": os.path.join(IMG_DIR, "Iris versicolor.jpg"),
    "virginica": os.path.join(IMG_DIR, "Iris virginica.jpg"),
}


# ============================================================
# 3. PHẦN HUẤN LUYỆN MÔ HÌNH (dùng cache: chỉ chạy 1 lần)
# ============================================================
@st.cache_resource
def load_model():
    """Huấn luyện GaussianNB trên toàn bộ Iris; đồng thời tính accuracy trên tập Test (80/20)."""
    iris = load_iris()
    X, y = iris.data, iris.target

    # Mô hình dùng cho demo: huấn luyện trên TOÀN BỘ dữ liệu (giống bản Gradio)
    model = GaussianNB()
    model.fit(X, y)

    # Mô hình phụ chỉ để đánh giá (bonus - hiển thị ở sidebar)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    eval_model = GaussianNB().fit(X_tr, y_tr)
    test_acc = accuracy_score(y_te, eval_model.predict(X_te))
    return model, iris, test_acc


# ============================================================
# 4. PHẦN DỰ ĐOÁN
# ============================================================
def predict_iris(model, iris, sl, sw, pl, pw):
    """Trả về (tên loài, mảng xác suất 3 lớp)."""
    x = np.array([[sl, sw, pl, pw]])
    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0]
    return iris.target_names[pred], proba


def load_species_image(species):
    """Trả về đường dẫn ảnh nếu tồn tại, ngược lại trả về None (không làm app bị dừng)."""
    path = SPECIES_IMAGES[species]
    return path if os.path.exists(path) else None


def plot_scatter(iris, user_point, species_pred, fx, fy):
    """Vẽ scatter plot dữ liệu Iris và đánh dấu điểm người dùng vừa nhập."""
    names = list(iris.feature_names)
    ix, iy = names.index(fx), names.index(fy)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    for i, sp in enumerate(iris.target_names):
        m = iris.target == i
        ax.scatter(iris.data[m, ix], iris.data[m, iy], c=colors[i], s=22, alpha=0.7, label=sp)
    ax.scatter(user_point[ix], user_point[iy], c="black", marker="*", s=300,
               edgecolors="yellow", linewidths=1.2, label=f"Điểm nhập ({species_pred})")
    ax.set_xlabel(fx)
    ax.set_ylabel(fy)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ============================================================
# 5. GIAO DIỆN
# ============================================================
def main():
    model, iris, test_acc = load_model()

    # ---------- Sidebar (bonus): thông tin mô hình ----------
    with st.sidebar:
        st.header("ℹ️ Thông tin mô hình")
        st.write("**Mô hình:** Gaussian Naive Bayes")
        st.write("**Dữ liệu:** Iris (150 mẫu, 4 đặc trưng, 3 loài)")
        st.write("**Chia dữ liệu:** 80/20, random_state=42, stratify=y")
        st.metric("Độ chính xác trên tập Test", f"{test_acc * 100:.2f}%")
        st.caption("Mô hình trong demo được huấn luyện trên toàn bộ dữ liệu.")

    # ---------- Tiêu đề ----------
    st.title("Demo Phân Loại Hoa Iris Dataset")
    st.subheader("Mô hình: Gaussian Naive Bayes")
    st.markdown("Nhập thông tin kích thước ở cột bên trái, bấm **Dự đoán ngay** "
                "để xem loài hoa tương ứng ở cột bên phải.")

    # ---------- Bố cục 2 cột ----------
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Nhập thông tin mẫu hoa")
        values = [
            st.slider(label, min_value=mn, max_value=mx, value=df, step=stp, key=f"slider_{i}")
            for i, (label, mn, mx, df, stp) in enumerate(SLIDERS)
        ]
        # st.button trả về True khi được bấm
        if st.button("🔍 Dự đoán ngay", type="primary"):
            species, proba = predict_iris(model, iris, *values)
            # Lưu kết quả để không mất khi trang chạy lại
            st.session_state["result"] = {"species": species, "proba": proba, "values": values}

    with col_right:
        st.markdown("### Kết quả dự đoán")
        result = st.session_state.get("result")
        if result is None:
            st.info("Chưa có kết quả. Hãy chọn giá trị và bấm **Dự đoán ngay**.")
        else:
            species, proba = result["species"], result["proba"]

            # Kết quả văn bản: in hoa, in đậm
            st.markdown(f"### Loài hoa được dự đoán: **{species.upper()}**")

            # Hình ảnh minh họa (xử lý lỗi thiếu file ảnh)
            img = load_species_image(species)
            if img:
                st.image(img, caption=f"Iris {species}", width=300)
            else:
                st.warning(f"Không tìm thấy ảnh minh họa cho loài '{species}' "
                           f"(cần file: hoa_Iris/Iris {species}.jpg).")

            # Xác suất 3 loài: thanh tiến trình
            st.markdown("**Xác suất tương ứng từng loài**")
            for name, p in zip(iris.target_names, proba):
                st.progress(float(p), text=f"{name.capitalize()}: {p * 100:.2f}%")

    # ---------- Bonus: scatter plot + bảng dữ liệu ----------
    st.divider()
    result = st.session_state.get("result")
    tab1, tab2 = st.tabs(["📊 Biểu đồ phân bố", "🗂️ Bảng dữ liệu Iris"])
    with tab1:
        if result is None:
            st.info("Bấm **Dự đoán ngay** để đánh dấu điểm vừa nhập lên biểu đồ.")
        else:
            c1, c2 = st.columns(2)
            fx = c1.selectbox("Trục X", iris.feature_names, index=2)
            fy = c2.selectbox("Trục Y", iris.feature_names, index=3)
            st.pyplot(plot_scatter(iris, result["values"], result["species"], fx, fy))
    with tab2:
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df["species"] = [iris.target_names[t] for t in iris.target]
        st.dataframe(df)


if __name__ == "__main__":
    main()
