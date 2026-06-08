# IMPORT THƯ VIỆN
# =========================================================
import streamlit as st
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

[data-testid="stHeader"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
st.markdown("""
<style>

.stApp{
    background-color:#071028;
    color:white;
}

h1,h2,h3,h4,h5,p,div{
    color:white;
}

[data-testid="stSidebar"]{
    background-color:#111c3a;
}

</style>
""", unsafe_allow_html=True)
st.title("📊 Bài 6—TOPSIS - xếp hạng 6 vùng kinh tế Việt Nam theo mức độ ưu tiên đầu tư AI ")
# =========================================================
# SIDEBAR ĐIỀU CHỈNH THAM SỐ
# =========================================================
st.sidebar.header("⚙️ Điều chỉnh TOPSIS")

w_grdp = st.sidebar.slider("GRDP/người", 0.00, 1.00, 0.10, 0.01)
w_fdi = st.sidebar.slider("FDI", 0.00, 1.00, 0.10, 0.01)
w_digital = st.sidebar.slider("Digital Index", 0.00, 1.00, 0.15, 0.01)
w_ai = st.sidebar.slider("AI Readiness", 0.00, 1.00, 0.20, 0.01)
w_labor = st.sidebar.slider("LĐ qua đào tạo", 0.00, 1.00, 0.15, 0.01)
w_rd = st.sidebar.slider("R&D/GRDP", 0.00, 1.00, 0.15, 0.01)
w_internet = st.sidebar.slider("Internet", 0.00, 1.00, 0.05, 0.01)
w_gini = st.sidebar.slider("Gini", 0.00, 1.00, 0.10, 0.01)

normalize_weights = st.sidebar.checkbox(
    "Chuẩn hóa tổng trọng số = 1",
    value=True
)

gini_as_cost = st.sidebar.checkbox(
    "Xem Gini là tiêu chí chi phí",
    value=True
)

top_n = st.sidebar.slider(
    "Số vùng top hiển thị",
    1, 6, 3, 1
)

st.sidebar.subheader("Độ nhạy AI Readiness")

ai_min = st.sidebar.slider("w_AI nhỏ nhất", 0.05, 0.40, 0.10, 0.05)
ai_max = st.sidebar.slider("w_AI lớn nhất", 0.10, 0.60, 0.40, 0.05)
ai_step = st.sidebar.slider("Bước nhảy w_AI", 0.01, 0.10, 0.05, 0.01)

if ai_min > ai_max:
    st.error("w_AI nhỏ nhất không được lớn hơn w_AI lớn nhất.")
    st.stop()
# =========================================================
# TITLE
# =========================================================
st.header("6.1. Bối cảnh Việt Nam")

st.markdown("""
Theo Quyết định số 127/QĐ-TTg ngày 26/01/2021 về Chiến lược quốc gia về nghiên cứu, phát triển và ứng dụng AI
đến năm 2030, Việt Nam đặt mục tiêu trở thành trung tâm AI của ASEAN.

Ngân sách triển khai có hạn nên cần lựa chọn vùng nào để triển khai các trung tâm AI và sandbox dữ liệu trước.
Bài tập này áp dụng phương pháp TOPSIS để xếp hạng 6 vùng kinh tế - xã hội theo mức độ sẵn sàng AI.
""")
st.header("6.4. Phân tích TOPSIS cho 6 vùng kinh tế Việt Nam")

# =========================================================
# DỮ LIỆU 6 VÙNG
# =========================================================
regions = [
    "Trung du miền núi phía Bắc",
    "Đồng bằng sông Hồng",
    "Bắc Trung Bộ & DH miền Trung",
    "Tây Nguyên",
    "Đông Nam Bộ",
    "Đồng bằng sông Cửu Long"
]

# Các tiêu chí
criteria = [
    "GRDP/người",
    "FDI",
    "Digital Index",
    "AI Readiness",
    "LĐ qua đào tạo",
    "R&D/GRDP",
    "Internet",
    "Gini"
]

# Dữ liệu từ đề bài
X = np.array([
    [57.0, 3.5, 38, 22, 21.5, 0.18, 72, 0.405],
    [152.3, 20.0, 78, 68, 36.8, 0.85, 92, 0.358],
    [87.5, 8.2, 55, 40, 27.5, 0.32, 84, 0.372],
    [68.9, 0.8, 32, 18, 18.2, 0.15, 68, 0.412],
    [158.9, 18.5, 82, 75, 42.5, 0.78, 94, 0.385],
    [80.5, 2.1, 48, 30, 16.8, 0.22, 78, 0.392]
])

# True = lợi ích
# False = chi phí
is_benefit = [
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    not gini_as_cost
]

# =========================================================
# DATAFRAME GỐC
# =========================================================
df_raw = pd.DataFrame(X, columns=criteria, index=regions)

st.subheader("Bảng dữ liệu đầu vào")
st.dataframe(df_raw, use_container_width=True)

# =========================================================
# HÀM TOPSIS
# =========================================================
def topsis_method(X, weights, is_benefit):

    # -----------------------------
    # Bước 1: Chuẩn hóa vector
    # -----------------------------
    R = X / np.sqrt((X ** 2).sum(axis=0))

    # -----------------------------
    # Bước 2: Ma trận trọng số
    # -----------------------------
    V = R * weights

    # -----------------------------
    # Bước 3: Giải pháp lý tưởng
    # -----------------------------
    A_star = np.zeros(X.shape[1])
    A_neg = np.zeros(X.shape[1])

    for j in range(X.shape[1]):

        if is_benefit[j]:
            A_star[j] = V[:, j].max()
            A_neg[j] = V[:, j].min()
        else:
            A_star[j] = V[:, j].min()
            A_neg[j] = V[:, j].max()

    # -----------------------------
    # Bước 4: Khoảng cách
    # -----------------------------
    S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))

    # -----------------------------
    # Bước 5: Hệ số gần tối ưu
    # -----------------------------
    C_star = S_neg / (S_star + S_neg)

    return R, V, A_star, A_neg, S_star, S_neg, C_star

# =========================================================
# 6.4.1 TOPSIS VỚI TRỌNG SỐ CHUYÊN GIA
# =========================================================
st.subheader("6.4.1. TOPSIS với trọng số chuyên gia")

weights_expert = np.array([
    w_grdp,
    w_fdi,
    w_digital,
    w_ai,
    w_labor,
    w_rd,
    w_internet,
    w_gini
])

if normalize_weights:
    total_weight = weights_expert.sum()

    if total_weight == 0:
        st.error("Tổng trọng số bằng 0. Hãy tăng ít nhất một trọng số.")
        st.stop()

    weights_expert = weights_expert / total_weight
df_weights = pd.DataFrame({
    "Tiêu chí": criteria,
    "Trọng số đang dùng": weights_expert
})

st.write("### Trọng số TOPSIS đang dùng")
st.dataframe(
    df_weights.style.format({
        "Trọng số đang dùng": "{:.4f}"
    }),
    use_container_width=True
)

R, V, A_star, A_neg, S_star, S_neg, C_star = topsis_method(
    X,
    weights_expert,
    is_benefit
)

df_topsis = pd.DataFrame({
    "Vùng": regions,
    "S+": S_star,
    "S-": S_neg,
    "C*": C_star
})

df_topsis["Xếp hạng"] = (
    df_topsis["C*"]
    .rank(ascending=False, method="dense")
    .astype(int)
)

df_topsis = df_topsis.sort_values("C*", ascending=False)
df_topsis = df_topsis.sort_values(
    "C*",
    ascending=False
)

# =====================================
# TOP N VÙNG ƯU TIÊN
# =====================================
st.write(f"### Top {top_n} vùng ưu tiên")

st.dataframe(
    df_topsis.head(top_n),
    use_container_width=True
)

st.write("### Kết quả TOPSIS")
st.write("### Kết quả TOPSIS")

st.dataframe(
    df_topsis.style.format({
        "S+": "{:.4f}",
        "S-": "{:.4f}",
        "C*": "{:.4f}"
    }),
    use_container_width=True
)
# =========================================================
# BIỂU ĐỒ TOPSIS
# =========================================================
fig, ax = plt.subplots(figsize=(10,5))

ax.bar(df_topsis["Vùng"], df_topsis["C*"])

plt.xticks(rotation=15)

ax.set_ylabel("TOPSIS Score")
ax.set_title("Xếp hạng TOPSIS theo trọng số chuyên gia")

st.pyplot(fig)

# =========================================================
# 6.4.2 ENTROPY
# =========================================================
st.subheader("6.4.2. Tính trọng số bằng phương pháp Entropy")

# =========================================================
# CHUẨN HÓA ENTROPY
# =========================================================
X_entropy = X.copy()

# Xử lý tiêu chí chi phí (Gini)
for j in range(X_entropy.shape[1]):

    if not is_benefit[j]:
        X_entropy[:, j] = 1 / X_entropy[:, j]

# =========================================================
# TÍNH ENTROPY WEIGHT
# =========================================================
P = X_entropy / X_entropy.sum(axis=0)

k = 1 / np.log(len(X_entropy))

E = -k * np.sum(
    P * np.log(P + 1e-12),
    axis=0
)

d = 1 - E

weights_entropy = d / d.sum()

# =========================================================
# HIỂN THỊ TRỌNG SỐ ENTROPY
# =========================================================
df_entropy_weight = pd.DataFrame({
    "Tiêu chí": criteria,
    "Trọng số Entropy": weights_entropy
})

st.dataframe(
    df_entropy_weight.style.format({
        "Trọng số Entropy": "{:.4f}"
    }),
    use_container_width=True
)

# =========================================================
# TOPSIS VỚI ENTROPY
# =========================================================
_, _, _, _, S_star_e, S_neg_e, C_star_e = topsis_method(
    X,
    weights_entropy,
    is_benefit
)

df_entropy = pd.DataFrame({
    "Vùng": regions,
    "C*_Entropy": C_star_e
})

df_entropy["Rank_Entropy"] = (
    df_entropy["C*_Entropy"]
    .rank(ascending=False, method="dense")
    .astype(int)
)

df_entropy = df_entropy.sort_values(
    "C*_Entropy",
    ascending=False
)

st.write("### Kết quả TOPSIS với Entropy")

st.dataframe(
    df_entropy.style.format({
        "C*_Entropy": "{:.4f}"
    }),
    use_container_width=True
)

# =========================================================
# SO SÁNH XẾP HẠNG
# =========================================================
st.subheader("So sánh xếp hạng giữa Expert Weight và Entropy")

compare_df = pd.merge(
    df_topsis[["Vùng", "C*", "Xếp hạng"]],
    df_entropy[["Vùng", "C*_Entropy", "Rank_Entropy"]],
    on="Vùng"
)

compare_df["Chênh lệch hạng"] = (
    compare_df["Rank_Entropy"]
    - compare_df["Xếp hạng"]
)

st.dataframe(
    compare_df.style.format({
        "C*": "{:.4f}",
        "C*_Entropy": "{:.4f}"
    }),
    use_container_width=True
)

# =========================================================
# 6.4.3 PHÂN TÍCH ĐỘ NHẠY w_AI
# =========================================================
st.subheader("6.4.3. Phân tích độ nhạy của trọng số AI")

ai_weights = np.arange(ai_min, ai_max + ai_step, ai_step)

sensitivity_result = []

for w_ai in ai_weights:

    w_temp = weights_expert.copy()
    w_temp[3] = w_ai
    w_temp = w_temp / w_temp.sum()


    

    _, _, _, _, _, _, C_temp = topsis_method(
        X,
        w_temp,
        is_benefit
    )

    df_temp = pd.DataFrame({
        "Vùng": regions,
        "Score": C_temp
    })

    top3 = (
        df_temp
        .sort_values("Score", ascending=False)
        .head(3)["Vùng"]
        .tolist()
    )

    sensitivity_result.append({
        "w_AI": round(w_ai, 2),
        "Top 3": " | ".join(top3)
    })

df_sensitivity = pd.DataFrame(sensitivity_result)

st.dataframe(df_sensitivity, use_container_width=True)
# =========================================================
# PHÂN TÍCH BỔ SUNG CHO 6.4.3
# =========================================================

st.markdown("""
### Phân tích kết quả độ nhạy

Khi trọng số AI Readiness tăng từ 0.10 lên 0.40,
thứ hạng giữa các vùng có thay đổi nhưng TOP-3 nhìn chung vẫn khá ổn định.

Các vùng có:
- Hạ tầng số mạnh
- AI readiness cao
- Và năng lực đổi mới sáng tạo tốt

vẫn duy trì lợi thế rõ rệt trong xếp hạng TOPSIS.

Điều này cho thấy các trung tâm công nghệ lớn hiện nay
không chỉ mạnh riêng về AI mà còn có lợi thế tổng hợp về:
FDI, lao động chất lượng cao, R&D và Internet.

### Hàm ý chính sách

Nếu Chính phủ tiếp tục ưu tiên AI trong chiến lược phát triển,
nguồn lực sẽ ngày càng tập trung vào các vùng công nghệ mạnh,
đồng thời làm gia tăng khoảng cách số giữa các vùng.
""")

# =========================================================
# 6.4.4 AHP ĐƠN GIẢN
# =========================================================
st.subheader("6.4.4. So sánh với AHP đơn giản")

ahp_weights = weights_expert.copy()

_, _, _, _, _, _, C_ahp = topsis_method(
    X,
    ahp_weights,
    is_benefit
)

df_ahp = pd.DataFrame({
    "Vùng": regions,
    "AHP Score": C_ahp
})

df_ahp = df_ahp.sort_values(
    "AHP Score",
    ascending=False
)

st.dataframe(
    df_ahp.style.format({
        "AHP Score": "{:.4f}"
    }),
    use_container_width=True
)


# =========================================================
# 6.5 THẢO LUẬN
# =========================================================



st.header("6.5. Câu hỏi thảo luận chính sách")

# =========================================================
# =========================================================
# 6.5. CÂU HỎI THẢO LUẬN CHÍNH SÁCH
# =========================================================
# =========================================================
# 6.5. CÂU HỎI THẢO LUẬN CHÍNH SÁCH
# =========================================================

# =========================================================
# DỮ LIỆU TỪ KẾT QUẢ TOPSIS
# =========================================================

top_region = df_topsis.iloc[0]["Vùng"]
top_score = df_topsis.iloc[0]["C*"]

# So sánh thứ hạng Expert vs Entropy
compare_df = pd.merge(
    df_topsis[["Vùng", "Xếp hạng"]],
    df_entropy[["Vùng", "Rank_Entropy"]],
    on="Vùng"
)

compare_df["Rank_Diff"] = abs(
    compare_df["Xếp hạng"] - compare_df["Rank_Entropy"]
)

max_change_region = compare_df.sort_values(
    "Rank_Diff",
    ascending=False
).iloc[0]

# Top 3 vùng
top3_regions = (
    df_topsis
    .sort_values("C*", ascending=False)
    .head(3)["Vùng"]
    .tolist()
)

# =========================================================
# CÂU A
# =========================================================

st.subheader(
    "a) Vùng nào dẫn đầu theo TOPSIS với trọng số chuyên gia? "
    "Đây có phải vùng nên triển khai trung tâm AI quốc gia đầu tiên không?"
)

st.markdown(f"""
Kết quả TOPSIS với bộ trọng số chuyên gia cho thấy vùng đứng đầu là:

### 🏆 {top_region}

với hệ số gần tối ưu:

### C* = {top_score:.4f}

Điều này phản ánh rằng vùng này có sự kết hợp tốt nhất giữa:
- GRDP bình quân đầu người
- Khả năng thu hút FDI
- Mức độ sẵn sàng AI
- Năng lực R&D
- Chất lượng lao động
- Và hạ tầng Internet.

### Phân tích chính sách

Đây là ứng viên phù hợp để triển khai trung tâm AI quốc gia đầu tiên vì:

- Có hệ sinh thái công nghệ mạnh
- Khả năng hấp thụ công nghệ AI cao
- Cơ sở hạ tầng số phát triển
- Và khả năng thu hút doanh nghiệp công nghệ lớn.

Ngoài ra, việc triển khai trung tâm AI tại vùng dẫn đầu
có thể tạo hiệu ứng lan tỏa công nghệ sang các vùng khác,
thúc đẩy đổi mới sáng tạo trên phạm vi toàn quốc.

Tuy nhiên, kết quả TOPSIS chủ yếu phản ánh hiệu quả kinh tế - công nghệ.
Trong thực tế, Chính phủ còn cần xem xét thêm:

- Yếu tố an ninh dữ liệu,
- Khả năng phân tán rủi ro,
- Cân bằng phát triển vùng miền,
- Và chiến lược địa - chính trị dài hạn.

Do đó, vùng đứng đầu TOPSIS nên được xem là
“ứng viên tối ưu về mặt kinh tế”,
nhưng quyết định cuối cùng cần kết hợp thêm các mục tiêu chiến lược quốc gia.
""")

# =========================================================
# CÂU B
# =========================================================

st.subheader(
    "b) Khi dùng trọng số Entropy, vùng nào có sự thay đổi xếp hạng lớn nhất? Vì sao?"
)

st.markdown(f"""
Khi chuyển từ trọng số chuyên gia sang trọng số Entropy,
vùng có mức thay đổi thứ hạng lớn nhất là:

### 📌 {max_change_region['Vùng']}

Mức thay đổi thứ hạng:

### ΔRank = {int(max_change_region['Rank_Diff'])}

### Giải thích nguyên nhân

Phương pháp Entropy xác định trọng số hoàn toàn dựa trên dữ liệu thực tế,
không phụ thuộc vào đánh giá chủ quan của chuyên gia.

Các tiêu chí có độ phân tán lớn giữa các vùng
sẽ được gán trọng số cao hơn,
vì chúng chứa nhiều “thông tin phân biệt” hơn.

Do đó:
- Nếu một vùng mạnh ở các tiêu chí được Entropy tăng trọng số,
  vùng đó sẽ tăng hạng;
- Ngược lại, nếu lợi thế của vùng nằm ở các tiêu chí bị giảm trọng số,
  thứ hạng sẽ giảm đáng kể.

### Ý nghĩa chính sách

Kết quả này cho thấy:
thứ hạng phát triển AI và chuyển đổi số
phụ thuộc khá mạnh vào cách lựa chọn trọng số.

Nếu Chính phủ ưu tiên:
- AI readiness
- Đổi mới sáng tạo
- Và R&D

thì kết quả xếp hạng sẽ khác với trường hợp ưu tiên:
- Phổ cập Internet
- Cân bằng xã hội
- Hoặc hạ tầng cơ bản.

Vì vậy, việc xác định trọng số trong các mô hình MCDM
cần minh bạch,đồng thời phản ánh đúng ưu tiên chiến lược quốc gia.
""")

# =========================================================
# CÂU C
# =========================================================

st.subheader(
    "c) TOPSIS giả định độc lập tuyến tính giữa các tiêu chí. "
    "Trong thực tế, AI Readiness và Internet penetration có thể tương quan rất cao. "
    "Điều này ảnh hưởng đến kết quả như thế nào? Đề xuất cách xử lý."
)

st.markdown("""
Một giả định quan trọng của TOPSIS là:
các tiêu chí hoạt động độc lập với nhau.

Tuy nhiên, trong thực tế,
AI Readiness và Internet penetration
thường có tương quan dương rất mạnh.

Các vùng có:
- Hạ tầng Internet phát triển
- Mức độ kết nối cao
- Và phổ cập số tốt

thường cũng là các vùng:
- Có hệ sinh thái AI mạnh
- Năng lực công nghệ cao
- Khả năng ứng dụng AI tốt hơn.

### Ảnh hưởng tới kết quả TOPSIS

Khi hai tiêu chí tương quan cao cùng xuất hiện,
một số vùng có thể được “cộng điểm hai lần”.

Điều này dẫn tới:
- Thiên lệch xếp hạng
- Khuếch đại lợi thế của các trung tâm phát triển mạnh
- Làm giảm cơ hội của các vùng đang phát triển.

Nói cách khác,mô hình có thể vô tình đếm lặp cùng một lợi thế công nghệ.

### Đề xuất xử lý

Có thể áp dụng một số giải pháp:

#### 1. Kiểm tra ma trận tương quan
Loại bỏ bớt các tiêu chí có tương quan quá cao.

#### 2. Sử dụng PCA
Gộp các tiêu chí tương quan mạnh
thành một nhân tố tổng hợp.

#### 3. Điều chỉnh trọng số
Giảm vai trò của các biến trùng lặp thông tin.

#### 4. Sử dụng mô hình MCDM nâng cao
Ví dụ:
- ANP,
- DEMATEL,
- Fuzzy TOPSIS

để mô hình hóa quan hệ phụ thuộc giữa các tiêu chí.

### Hàm ý chính sách

Nếu không xử lý tương quan giữa các tiêu chí,
nguồn lực AI quốc gia có thể bị tập trung quá mức
vào một vài cực tăng trưởng,
làm gia tăng chênh lệch phát triển vùng miền.
""")

# =========================================================
# CÂU D
# =========================================================

st.subheader(
    "d) Theo Quyết định 127/QĐ-TTg, Việt Nam đặt mục tiêu xây dựng "
    "3 trung tâm AI lớn. Em sẽ chọn 3 vùng nào dựa trên kết quả TOPSIS? "
    "Có cần điều chỉnh thêm tiêu chí địa - chính trị không?"
)

st.markdown(f"""
Dựa trên kết quả TOPSIS,
3 vùng phù hợp nhất để phát triển trung tâm AI quốc gia gồm:

### 🥇 {top3_regions[0]}
### 🥈 {top3_regions[1]}
### 🥉 {top3_regions[2]}

### Cơ sở lựa chọn

Các vùng này có:
- AI Readiness cao
- Hạ tầng số phát triển
- Năng lực R&D mạnh
- Tỷ lệ lao động qua đào tạo lớn
- Và khả năng thu hút FDI tốt hơn các vùng còn lại.

Ngoài ra:
- Mức độ phổ cập Internet
- Quy mô kinh tế số
- Và khả năng hấp thụ công nghệ
cũng vượt trội hơn đáng kể.

### Tuy nhiên, chỉ dùng TOPSIS là chưa đủ

TOPSIS chủ yếu tối ưu hiệu quả kinh tế và công nghệ.
Trong thực tế,quy hoạch trung tâm AI quốc gia còn liên quan tới:

- An ninh quốc gia
- Chiến lược Bắc - Trung - Nam
- Khả năng chống chịu thiên tai
- An toàn dữ liệu
- Và mục tiêu phát triển cân bằng vùng miền.

### Đề xuất điều chỉnh địa - chính trị

Chính phủ nên cân nhắc:
- Ít nhất một trung tâm AI ở miền Bắc
- Một ở miền Nam
- Và một trung tâm chiến lược tại miền Trung hoặc Tây Nguyên.

Cách tiếp cận này giúp:
- Giảm rủi ro tập trung hạ tầng,
- Tăng khả năng dự phòng chiến lược,
- Và tạo hiệu ứng lan tỏa công nghệ trên phạm vi toàn quốc.

### Kết luận

TOPSIS là công cụ định lượng hữu ích
để xác định vùng ưu tiên phát triển AI.

Tuy nhiên,quyết định cuối cùng cần kết hợp:
- Hiệu quả kinh tế
- An ninh chiến lược
- Và mục tiêu phát triển bền vững quốc gia.
""")

st.success(" Hoàn thành bài 6.")
# =========================================================
# OUTPUT CHO BÀI 12 - AIDEOM VN
# =========================================================

st.markdown("---")
st.header("📤 Output cho AIDEOM-VN Dashboard")

# ============================================
# OUTPUT MODULE M2
# ============================================

M2_OUTPUT = {

    "best_region": top_region,

    "digital_index": round(top_score * 100, 2),

    "top3_regions": top3_regions,

    "ai_readiness_score": round(
        df_topsis["C*"].mean() * 100,
        2
    )
}

# ============================================
# HIỂN THỊ
# ============================================


def get_m2_output():
    return M2_OUTPUT


if __name__ == "__main__":
    st.json(get_m2_output())

st.success("""
✅ M2 Output đã sẵn sàng để tích hợp vào:
Bài 12 — AIDEOM-VN Dashboard
""")
