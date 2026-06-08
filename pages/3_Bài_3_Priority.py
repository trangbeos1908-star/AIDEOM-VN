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
import plotly.express as px

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="Bài 3 - Priority Index",
    layout="wide"
)
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
# ==================================================
# TITLE
# ==================================================
st.title("📊 Bài 3 — Priority Index cho 10 ngành Việt Nam")
# ==================================================
# SIDEBAR ĐIỀU CHỈNH THAM SỐ
# ==================================================
st.sidebar.header("⚙️ Điều chỉnh tham số bài 3")

st.sidebar.subheader("Trọng số Priority Index")

w_growth = st.sidebar.slider("a1 - Growth", 0.0, 1.0, 0.15, 0.01)
w_productivity = st.sidebar.slider("a2 - Productivity", 0.0, 1.0, 0.15, 0.01)
w_spillover = st.sidebar.slider("a3 - Spillover", 0.0, 1.0, 0.20, 0.01)
w_export = st.sidebar.slider("a4 - Export", 0.0, 1.0, 0.15, 0.01)
w_employment = st.sidebar.slider("a5 - Employment", 0.0, 1.0, 0.10, 0.01)
w_ai = st.sidebar.slider("a6 - AI Readiness", 0.0, 1.0, 0.20, 0.01)
w_risk = st.sidebar.slider("a7 - Risk", 0.0, 1.0, 0.15, 0.01)

normalize_weights = st.sidebar.checkbox(
    "Chuẩn hóa tổng trọng số = 1",
    value=True
)

top_n = st.sidebar.slider(
    "Số ngành hiển thị trong bảng xếp hạng",
    min_value=3,
    max_value=10,
    value=10,
    step=1
)

st.sidebar.subheader("Độ nhạy AI Readiness")

ai_min = st.sidebar.slider("a6 nhỏ nhất", 0.05, 0.40, 0.05, 0.05)
ai_max = st.sidebar.slider("a6 lớn nhất", 0.05, 0.60, 0.40, 0.05)
ai_step = st.sidebar.slider("Bước nhảy a6", 0.01, 0.10, 0.05, 0.01)

# ==================================================
# LOAD DATA
# ==================================================
# Đọc file dữ liệu từ thư mục
df = pd.read_csv("vietnam_sectors_2024 (1).csv")
# ==================================================
# 3.1 BỐI CẢNH VIỆT NAM
# ==================================================

st.header("3.1. Bối cảnh Việt Nam")

st.markdown("""
Theo cơ cấu kinh tế năm 2024 do Cục Thống kê quốc gia công bố:

- Nông - lâm - thủy sản chiếm **11,86% GDP**
- Công nghiệp - xây dựng chiếm **37,64% GDP**
- Dịch vụ chiếm **42,36% GDP**

Trong bối cảnh chuyển đổi số và trí tuệ nhân tạo (AI)
đang trở thành động lực tăng trưởng mới,
Việt Nam đối mặt với câu hỏi chính sách quan trọng:

> Trong số các ngành kinh tế lớn,
> ngành nào nên được ưu tiên đẩy mạnh
> chuyển đổi số và ứng dụng AI trước
> để tạo hiệu ứng lan tỏa tối đa cho nền kinh tế?

Do nguồn lực đầu tư là hữu hạn,
việc lựa chọn ngành ưu tiên cần dựa trên
một chỉ số định lượng khách quan.

Vì vậy,
nghiên cứu xây dựng một chỉ số mang tên:

$$
Priority_i
$$

nhằm đánh giá mức độ ưu tiên chuyển đổi số
của từng ngành kinh tế Việt Nam.
""")

# ==================================================
# 3.2 MÔ HÌNH TOÁN HỌC
# ==================================================

st.header("3.2. Mô hình toán học")

st.markdown(r"""
Theo mô hình nghiên cứu,
chỉ số ưu tiên ngành $i$ được xác định:

$$
Priority_i =
a_1 Growth_i
+
a_2 Productivity_i
+
a_3 Spillover_i
+
a_4 Export_i
+
a_5 Employment_i
+
a_6 AIReadiness_i
-
a_7 Risk_i
$$

Trong đó:

- $Growth_i$: Tăng trưởng ngành
- $Productivity_i$: Năng suất lao động
- $Spillover_i$: Mức độ lan tỏa
- $Export_i$: Năng lực xuất khẩu
- $Employment_i$: Quy mô việc làm
- $AIReadiness_i$: Mức độ sẵn sàng AI
- $Risk_i$: Rủi ro tự động hóa

---

# Chuẩn hóa dữ liệu Min-Max

Trước khi tính chỉ số Priority,
các biến cần được chuẩn hóa
về thang đo:

$$
[0,1]
$$

để loại bỏ khác biệt đơn vị đo lường.

## Với các chỉ số tốt

$$
\tilde{x}_i
=
\frac{x_i - \min(x)}
{\max(x)-\min(x)}
$$

## Với chỉ số xấu như Risk

$$
\tilde{x}_i
=
\frac{\max(x)-x_i}
{\max(x)-\min(x)}
$$

Việc đảo dấu giúp:

- Ngành có rủi ro thấp được điểm cao hơn
- Phản ánh đúng mục tiêu ưu tiên chính sách
- Hạn chế tác động tiêu cực của tự động hóa
""")




# ==================================================
# 3.4.1 CHUẨN HÓA DỮ LIỆU
# ==================================================
st.subheader("3.4.1 Chuẩn hóa dữ liệu Min-Max")

cols_good = [
    'growth_rate_2024_pct',
    'gdp_share_2024_pct',
    'spillover_coef_0_1',
    'export_billion_USD',
    'labor_million',
    'ai_readiness_0_100'
]
col_bad = 'automation_risk_pct'

def norm_good(x):
    return (x - x.min()) / (x.max() - x.min())

def norm_bad(x):
    return (x.max() - x) / (x.max() - x.min())

Xg = df[cols_good].apply(norm_good)
Xb = norm_bad(df[col_bad])

# Ma trận đã chuẩn hóa gồm cả 7 cột
normalized_matrix = pd.concat([Xg, Xb], axis=1)

normalized_df = pd.DataFrame({
    "Ngành": df["sector_name_vi"],
    "Growth": Xg['growth_rate_2024_pct'],
    "Productivity": Xg['gdp_share_2024_pct'],
    "Spillover": Xg['spillover_coef_0_1'],
    "Export": Xg['export_billion_USD'],
    "Employment": Xg['labor_million'],
    "AI Readiness": Xg['ai_readiness_0_100'],
    "Risk (Adjusted)": Xb
})

st.markdown("**Ma trận dữ liệu đã chuẩn hóa:**")
st.dataframe(normalized_df, use_container_width=True)

# ==================================================
# 3.4.2 TÍNH PRIORITY VÀ XẾP HẠNG
# ==================================================
st.subheader("3.4.2 Tính Priority Index và xếp hạng ngành")

# Trọng số mặc định đề bài cho: 6 cột tốt và 1 cột xấu (Risk)
w_all = np.array([
    w_growth,
    w_productivity,
    w_spillover,
    w_export,
    w_employment,
    w_ai,
    w_risk
])

if normalize_weights:
    total_w = w_all.sum()
    if total_w == 0:
        st.error("Tổng trọng số đang bằng 0. Hãy tăng ít nhất một trọng số.")
        st.stop()
    w_all = w_all / total_w

w_default = w_all[:6]
w_risk_default = w_all[6]

# ĐÚNG CÔNG THỨC: Dấu TRỪ theo gợi ý mã nguồn của đề bài
priority = Xg.values @ w_default - w_risk_default * Xb.values
df['Priority'] = priority
ranking = df.sort_values(
    'Priority',
    ascending=False
)[['sector_name_vi', 'Priority']].head(top_n).reset_index(drop=True)
ranking.index = ranking.index + 1

st.dataframe(ranking, use_container_width=True)

fig_rank = px.bar(
    ranking,
    x='sector_name_vi',
    y='Priority',
    title='Xếp hạng Priority Index các ngành (Mặc định)',
    text_auto='.3f'
)
st.plotly_chart(fig_rank, use_container_width=True)

# ==================================================
# 3.4.3 PHÂN TÍCH ĐỘ NHẠY & HEATMAP
# ==================================================
st.subheader("3.4.3 Phân tích độ nhạy theo AI Readiness (a6)")

# Chạy chính xác từ 0.05 đến 0.40, bước nhảy 0.05
if ai_min > ai_max:
    st.error("a6 nhỏ nhất không được lớn hơn a6 lớn nhất.")
    st.stop()

ai_weights = np.arange(ai_min, ai_max + ai_step, ai_step)

heatmap_records = []
all_sectors = df['sector_name_vi'].tolist()

# Định nghĩa bộ trọng số gốc cho 7 cột để dễ tính tỷ lệ chuẩn hóa lại
# Thứ tự: [growth, productivity, spillover, export, labor, ai_readiness, risk]
w_base_7 = w_all.copy()

for a6 in ai_weights:
    # Sao chép bộ trọng số gốc
    w_temp = w_base_7.copy()
    w_temp[5] = a6 # Thay đổi vị trí của AI Readiness (a6)
    
    # Chuẩn hóa lại các trọng số còn lại sao cho tổng bằng 1
    other_indices = [0, 1, 2, 3, 4, 6]
    remaining_sum = w_temp[other_indices].sum()
    w_temp[other_indices] = w_temp[other_indices] * ((1 - a6) / remaining_sum)
    
    # Tách trọng số tốt và trọng số risk sau khi đã chuẩn hóa
    w_g_temp = w_temp[0:6]
    w_b_temp = w_temp[6]
    
    # Tính lại Priority chuẩn công thức
    p_temp = Xg.values @ w_g_temp - w_b_temp * Xb.values
    
    temp_df = df.copy()
    temp_df['Priority_Temp'] = p_temp
    # Xếp hạng từ cao xuống thấp
    temp_df = temp_df.sort_values('Priority_Temp', ascending=False).reset_index(drop=True)
    
    # Lưu kết quả Top 3 ngành dưới dạng text để hiển thị bảng
    top3_names = temp_df.head(3)['sector_name_vi'].tolist()
    heatmap_records.append({
        'AI Weight (a6)': f"a6 = {a6:.2f}",
        'Top 1': top3_names[0],
        'Top 2': top3_names[1],
        'Top 3': top3_names[2]
    })

sensitivity_df = pd.DataFrame(heatmap_records).set_index('AI Weight (a6)')
st.write("**Bảng theo dõi thay đổi Top-3 ngành theo trọng số AI:**")
st.dataframe(sensitivity_df, use_container_width=True)

# --- VẼ HEATMAP CHUẨN ---
# Để vẽ heatmap, ta tạo ma trận điểm/thứ hạng của TẤT CẢ các ngành theo sự thay đổi của a6
heatmap_matrix = []
for a6 in ai_weights:
    w_temp = w_base_7.copy()
    w_temp[5] = a6
    other_indices = [0, 1, 2, 3, 4, 6]
    remaining_sum = w_temp[other_indices].sum()
    w_temp[other_indices] = w_temp[other_indices] * ((1 - a6) / remaining_sum)
    
    p_temp = Xg.values @ w_temp[0:6] - w_temp[6] * Xb.values
    temp_df = df.copy()
    temp_df['Priority_Temp'] = p_temp
    # Tính thứ hạng (1 là cao nhất)
    temp_df['Rank'] = temp_df['Priority_Temp'].rank(ascending=False, method='min')
    heatmap_matrix.append(temp_df.sort_values('sector_name_vi')['Rank'].values)

sector_names_sorted = df.sort_values('sector_name_vi')['sector_name_vi'].tolist()
heatmap_plot_df = pd.DataFrame(
    heatmap_matrix, 
    index=[f"a6={x:.2f}" for x in ai_weights], 
    columns=sector_names_sorted
)

fig_heat = px.imshow(
    heatmap_plot_df,
    text_auto=True,
    labels=dict(x="Ngành kinh tế", y="Trọng số a6", color="Thứ hạng (Càng nhỏ càng ưu tiên)"),
    title="Heatmap Thay đổi thứ hạng của tất cả các ngành theo trọng số AI Readiness",
    color_continuous_scale="RdYlGn_r" # Màu đỏ là hạng thấp, xanh là hạng cao
)
st.plotly_chart(fig_heat, use_container_width=True)

# ==================================================
# 3.4.4 SO SÁNH HAI ĐỊNH HƯỚNG CHÍNH SÁCH
# ==================================================
# ==================================================
# 3.4.4 SO SÁNH HAI ĐỊNH HƯỚNG CHÍNH SÁCH (MỞ RỘNG)
# ==================================================

st.subheader("3.4.4 So sánh hai định hướng chính sách")

# --- BỘ TRỌNG SỐ 1: ĐỊNH HƯỚNG TĂNG TRƯỞNG ---
growth_weights = np.array([0.25, 0.20, 0.10, 0.25, 0.05, 0.10])
growth_risk = 0.05

df['Priority_Growth'] = Xg.values @ growth_weights - growth_risk * Xb.values
df['Rank_Growth'] = df['Priority_Growth'].rank(ascending=False, method='min').astype(int)

# --- BỘ TRỌNG SỐ 2: ĐỊNH HƯỚNG BAO TRÙM ---
inclusive_weights = np.array([0.10, 0.10, 0.25, 0.05, 0.25, 0.10])
inclusive_risk = 0.15

df['Priority_Inclusive'] = Xg.values @ inclusive_weights - inclusive_risk * Xb.values
df['Rank_Inclusive'] = df['Priority_Inclusive'].rank(ascending=False, method='min').astype(int)

# --- TÍNH TOÁN SỰ CHÊNH LỆCH THỨ HẠNG ---
# Sự thay đổi vị trí từ Tăng trưởng sang Bao trùm
df['Rank_Change'] = df['Rank_Growth'] - df['Rank_Inclusive'] 

# Hiển thị 2 cột song song
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📈 Định hướng Tăng trưởng (Ưu tiên Xuất khẩu, GDP, Năng suất)")
    rank_growth_disp = df.sort_values('Rank_Growth')[['Rank_Growth', 'sector_name_vi', 'Priority_Growth']]
    st.dataframe(rank_growth_disp.rename(columns={'Rank_Growth': 'Hạng', 'sector_name_vi': 'Ngành'}), use_container_width=True, hide_index=True)

with col2:
    st.markdown("#### 🤝 Định hướng Bao trùm (Ưu tiên Việc làm, Lan tỏa, Giảm rủi ro)")
    rank_inclusive_disp = df.sort_values('Rank_Inclusive')[['Rank_Inclusive', 'sector_name_vi', 'Priority_Inclusive']]
    st.dataframe(rank_inclusive_disp.rename(columns={'Rank_Inclusive': 'Hạng', 'sector_name_vi': 'Ngành'}), use_container_width=True, hide_index=True)

# --- BIỂU ĐỒ SO SÁNH SỰ DỊCH CHUYỂN THỨ HẠNG ---
st.markdown("#### 📊 Biểu đồ thay đổi thứ hạng khi chuyển từ Tăng trưởng sang Bao trùm")

# Chuẩn bị dữ liệu vẽ biểu đồ dạng thanh thể hiện sự thăng tiến/sụt giảm hạng
df_chart = df.sort_values('Rank_Change', ascending=True)
fig_compare = px.bar(
    df_chart,
    x='Rank_Change',
    y='sector_name_vi',
    orientation='h',
    title="Sự dịch chuyển thứ hạng (Dương: Thăng hạng khi đổi sang Bao trùm | Âm: Tụt hạng)",
    labels={'Rank_Change': 'Số bậc thay đổi thứ hạng', 'sector_name_vi': 'Ngành kinh tế'},
    color='Rank_Change',
    color_continuous_scale='RdYlGn'
)
st.plotly_chart(fig_compare, use_container_width=True)

# --- PHẦN NHẬN XÉT CHI TIẾT VÀ CHUYÊN SÂU ---
st.markdown("""
### 📝 Nhận xét phân tích chính sách chuyên sâu

Việc thay đổi cấu trúc trọng số giữa hai kịch bản đã dẫn đến sự phân hóa và dịch chuyển thứ hạng vô cùng rõ rệt giữa các nhóm ngành, minh chứng cho việc **chính sách công định hướng bằng dữ liệu (Data-driven Policy)** sẽ thay đổi thực tế phân bổ nguồn lực như thế nào:

1. **Các ngành bứt phá mạnh mẽ trong "Định hướng Bao trùm":**
    * **Nông - Lâm - Thủy sản và Bán buôn - bán lẻ:** Đây là các ngành chứng kiến sự thăng hạng vượt trội nhất. Nguyên nhân do bộ trọng số Bao trùm đã đẩy mạnh chỉ số **Labor** (Việc làm - từ 0.05 lên 0.25) và khả năng **Spillover** (Lan tỏa). Vì đây là những ngành đang giữ cấu trúc lao động rất lớn tại Việt Nam, việc chuyển đổi số ở đây mang ý nghĩa an sinh và giữ vững huyết mạch kinh tế cho số đông người dân hơn là chạy theo các con số tăng trưởng thuần túy.
    * **Rủi ro tự động hóa (Risk):** Trọng số rủi ro tăng từ 0.05 lên 0.15 khiến các ngành có tính an toàn cao (hoặc ít bị AI thay thế trực tiếp ở giai đoạn đầu) được bảo vệ và nâng đỡ trên bảng xếp hạng.

2. **Các ngành bị suy giảm vị thế tương đối trong "Định hướng Tăng trưởng":**
    * **Công nghiệp chế biến chế tạo và CNTT-Truyen thong:** Khi đặt vào kịch bản "Tăng trưởng", hai ngành này luôn độc chiếm vị trí Top đầu nhờ tỷ trọng xuất khẩu khổng lồ, đóng góp GDP trực tiếp lớn và tốc độ tăng trưởng phi mã. Tuy nhiên, khi chuyển sang "Bao trùm", vị trí của chúng bị đe dọa do thâm dụng lao động chất lượng cao nhưng quy mô tổng số lượng lao động chưa thể bao phủ diện rộng bằng khối dịch vụ truyền thống và nông nghiệp.

3. **Gợi ý lựa chọn chính sách (Policy Implications):**
    * **Không có bộ trọng số nào là hoàn hảo tuyệt đối:** Bộ trọng số "Tăng trưởng" giúp Việt Nam nhanh chóng xây dựng được các mũi nhọn kinh tế số để cạnh tranh sòng phẳng trên trường quốc tế. Trong khi đó, bộ "Bao trùm" lại là tấm lá chắn bảo vệ nền kinh tế khỏi nguy cơ phân hóa giàu nghèo và thất nghiệp công nghệ (Technological Unemployment).
    * **Chiến lược lai (Hybrid Strategy):** Khuyến nghị Chính phủ có thể áp dụng chiến lược theo giai đoạn. *Giai đoạn 1 (2024-2027):* Tập trung vào "Tăng trưởng" để tạo ra dòng vốn và hạ tầng công nghệ. *Giai đoạn 2 (2028-2030):* Dịch chuyển dần sang "Bao trùm" để phổ cập hóa công nghệ, đào tạo lại nguồn nhân lực diện rộng nhằm tránh các cú sốc xã hội do AI và tự động hóa mang lại.
""")

# ==================================================
# 3.5. THẢO LUẬN CHÍNH SÁCH
# ==================================================

st.header("3.5. Thảo luận chính sách")

# ==================================================
# CÂU A
# ==================================================

st.subheader(
    "a) Theo kết quả của em, ba ngành nào nên được ưu tiên đẩy mạnh chuyển đổi số và AI trước? Kết quả này có phù hợp với Nghị quyết 57-NQ/TW không?"
)

st.markdown(r"""
Theo kết quả tính toán Priority Index,
ba ngành có mức ưu tiên cao nhất thường bao gồm:

- CNTT - Truyền thông
- Công nghiệp chế biến chế tạo
- Tài chính - Ngân hàng

Đây là các ngành có:

- AI Readiness cao
- Khả năng lan tỏa mạnh
- Năng suất lao động lớn
- Tốc độ chuyển đổi số nhanh
- Khả năng tạo giá trị gia tăng cao

---

# Giải thích kinh tế

1. CNTT - Truyền thông

Đây là ngành đóng vai trò nền tảng
của toàn bộ quá trình chuyển đổi số quốc gia.
Ngành này hỗ trợ:
- Hạ tầng số
- Dữ liệu lớn
- Cloud computing
- AI platform
- Chính phủ số

Đầu tư vào CNTT giúp tạo hiệu ứng lan tỏa
tới hầu hết các ngành kinh tế khác.

---
 2. Công nghiệp chế biến chế tạo

Đây là khu vực xuất khẩu chủ lực của Việt Nam,
đồng thời có khả năng ứng dụng AI rất mạnh.

AI giúp:
- Tự động hóa sản xuất
- Tối ưu logistics
- Giảm chi phí vận hành
- Tăng năng suất lao động
- Nâng cao chất lượng sản phẩm

Do đó,ngành này tạo tác động lớn tới:
- GDP
- Xuất khẩu
- Năng lực cạnh tranh quốc gia

---
 3. Tài chính - Ngân hàng

Ngành tài chính có:
- Mức độ số hóa cao
- Dữ liệu lớn
- Khả năng triển khai AI nhanh

AI trong tài chính hỗ trợ:
- Phân tích dữ liệu khách hàng
- Chấm điểm tín dụng
- Phát hiện gian lận
- Tự động hóa dịch vụ

Đây là ngành có khả năng lan tỏa mạnh
tới tiêu dùng và đầu tư của toàn nền kinh tế.

# Liên hệ với Nghị quyết 57-NQ/TW

Kết quả này phù hợp với định hướng của
Nghị quyết 57-NQ/TW về:
- Đổi mới sáng tạo
- Chuyển đổi số quốc gia
- Phát triển khoa học công nghệ
- Làm chủ công nghệ chiến lược

Nghị quyết nhấn mạnh:
- Phát triển hạ tầng số
- Thúc đẩy AI
- Xây dựng kinh tế số
- Nâng cao năng suất quốc gia

Do đó, việc ưu tiên các ngành có:

- AI Readiness cao
- Hiệu ứng lan tỏa lớn
- Khả năng đổi mới công nghệ mạnh

là phù hợp với định hướng phát triển dài hạn của Việt Nam.
""")

# ==================================================
# CÂU B
# ==================================================

st.subheader(
    "b) Tại sao ngành Khai khoáng có năng suất rất cao nhưng vẫn không nằm trong nhóm ưu tiên?"
)

st.markdown(r"""
Mặc dù ngành Khai khoáng có năng suất lao động khá cao,
nhưng vẫn không đạt Priority Index cao
do tồn tại nhiều hạn chế cấu trúc.

# Các nguyên nhân chính

 1. Khả năng lan tỏa thấp

Khai khoáng chủ yếu khai thác tài nguyên thô,
ít tạo liên kết công nghệ mạnh
với các ngành khác trong nền kinh tế.

Do đó,
tác động lan tỏa của ngành không lớn.

 2. AI Readiness chưa cao

So với các ngành công nghệ hoặc tài chính,
khai khoáng có:

- Mức độ số hóa thấp hơn
- Hạ tầng dữ liệu hạn chế
- Khả năng ứng dụng AI chưa mạnh

Điều này làm giảm điểm ưu tiên
trong mô hình Priority Index.

3. Rủi ro tự động hóa cao

Nhiều công đoạn khai thác có thể:

- Tự động hóa
- Robot hóa
- Giảm nhu cầu lao động

Do đó,
Risk của ngành khá lớn,
khiến điểm Priority bị giảm.

 4. Tính bền vững dài hạn thấp

Khai khoáng phụ thuộc vào:

- Tài nguyên hữu hạn
- Giá hàng hóa thế giới
- Rủi ro môi trường

Trong khi đó,
mô hình ưu tiên ngành hướng tới:

- Tăng trưởng dài hạn
- Đổi mới công nghệ
- Phát triển bền vững

Vì vậy,
các ngành công nghệ cao thường được ưu tiên hơn.
""")

# ==================================================
# CÂU C
# ==================================================

st.subheader(
    "c) Bộ trọng số nên do ai quyết định: chuyên gia kỹ thuật, hội đồng chính sách hay đối thoại công khai?"
)

st.markdown(r"""
Việc xác định bộ trọng số trong Priority Index
là vấn đề rất quan trọng,
vì trọng số phản ánh mục tiêu phát triển quốc gia.

Nếu thay đổi trọng số,
thứ hạng ưu tiên ngành có thể thay đổi đáng kể.

1. Vai trò của chuyên gia kỹ thuật
Các chuyên gia dữ liệu và kinh tế có lợi thế về:
- Mô hình hóa định lượng
- Phân tích dữ liệu
- Đánh giá tác động chính sách
Họ giúp bảo đảm:
- Tính khoa học
- Tính logic
- Tính định lượng của mô hình
Tuy nhiên,
nếu chỉ phụ thuộc vào chuyên gia kỹ thuật, mô hình có thể thiếu góc nhìn xã hội và thực tiễn quản trị.

 2. Vai trò của hội đồng chính sách

Cơ quan hoạch định chính sách có khả năng:
- Liên kết với chiến lược quốc gia
- Cân đối tăng trưởng và an sinh
- Điều phối nguồn lực công

Tuy nhiên,quá trình này đôi khi chịu ảnh hưởng bởi:
- Mục tiêu nhiệm kỳ
- Lợi ích ngành
- Ưu tiên ngắn hạn


 3. Vai trò của đối thoại công khai

Tiếp cận hiện đại thường khuyến khích:
- Tham vấn doanh nghiệp
- Công khai dữ liệu
- Lấy ý kiến chuyên gia độc lập
- Đối thoại xã hội

Điều này giúp nâng cao:

- Tính minh bạch
- Tính chính danh chính sách
- Mức độ đồng thuận xã hội

# Kết luận

Trong thực tiễn,bộ trọng số nên được xây dựng theo mô hình kết hợp giữa:

- Chuyên gia kỹ thuật
- Hội đồng chính sách
- Đối thoại xã hội
để bảo đảm:
- Khoa học
- Minh bạch
- Khách quan
- Phù hợp mục tiêu phát triển dài hạn

Đây cũng là xu hướng governance hiện đại
trong hoạch định chính sách công dựa trên dữ liệu và bằng chứng.
""")
st.success("Hoàn thành toàn bộ Bài 3.")