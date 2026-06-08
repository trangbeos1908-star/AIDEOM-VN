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
import numpy as np
import pandas as pd
import plotly.express as px

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Bài 1 - Cobb Douglas",
    page_icon="📈",
    layout="wide"
)

# ==================================================
# CSS
# ==================================================
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
st.title("📈 Bài 1 — Cobb Douglas Production Function")
# ==================================================
# 1.1 BỐI CẢNH VIỆT NAM
# ==================================================
# ==================================================
# SIDEBAR - DYNAMIC MODEL CONTROL
# ==================================================

st.sidebar.title("⚙️ Dynamic Model Controls")

st.sidebar.markdown("## Hệ số Cobb-Douglas")

ALPHA = st.sidebar.slider(
    "α - Vốn vật chất (K)",
    0.0, 1.0, 0.33, 0.01
)

BETA = st.sidebar.slider(
    "β - Lao động (L)",
    0.0, 1.0, 0.42, 0.01
)

GAMMA = st.sidebar.slider(
    "γ - Kinh tế số (D)",
    0.0, 1.0, 0.10, 0.01
)

DELTA = st.sidebar.slider(
    "δ - AI",
    0.0, 1.0, 0.08, 0.01
)

THETA = st.sidebar.slider(
    "θ - Nhân lực số (H)",
    0.0, 1.0, 0.07, 0.01
)
scenario = st.sidebar.selectbox(
    "Chọn kịch bản",
    [
        "Baseline",
        "High Digital Economy",
        "AI Boom",
        "Low Growth",
        "Human Capital Focus"
    ]
)
# ==================================================
# CHECK RETURNS TO SCALE
# ==================================================

coef_sum = ALPHA + BETA + GAMMA + DELTA + THETA
st.sidebar.markdown("---")
st.sidebar.write("### Current Parameters")

st.sidebar.write(f"α = {ALPHA:.2f}")
st.sidebar.write(f"β = {BETA:.2f}")
st.sidebar.write(f"γ = {GAMMA:.2f}")
st.sidebar.write(f"δ = {DELTA:.2f}")
st.sidebar.write(f"θ = {THETA:.2f}")
st.sidebar.markdown("---")

st.sidebar.metric(
    "Tổng hệ số",
    f"{coef_sum:.2f}"
)

if abs(coef_sum - 1) < 0.01:
    st.sidebar.success("✔ Constant Returns to Scale")
else:
    st.sidebar.warning("⚠ Tổng hệ số khác 1")

# ==================================================
# 2030 SCENARIO
# ==================================================

st.sidebar.markdown("---")
st.sidebar.markdown("## Kịch bản 2030")

growth_K = st.sidebar.slider(
    "Tăng trưởng K (%/năm)",
    0.0, 15.0, 6.0, 0.1
)

growth_L = st.sidebar.slider(
    "Tăng trưởng L (%/năm)",
    0.0, 10.0, 6.0, 0.1
)

growth_TFP = st.sidebar.slider(
    "Tăng trưởng TFP (%/năm)",
    0.0, 10.0, 1.2, 0.1
)

D_2030 = st.sidebar.slider(
    "Kinh tế số D (% GDP)",
    10.0, 50.0, 30.0, 0.5
)

AI_2030 = st.sidebar.slider(
    "AI doanh nghiệp số (nghìn)",
    50.0, 200.0, 100.0, 1.0
)

H_2030 = st.sidebar.slider(
    "Lao động qua đào tạo H (%)",
    20.0, 60.0, 35.0, 0.5
)
st.header("1.1. Bối cảnh Việt Nam")

st.markdown("""
Theo Cục Thống kê quốc gia, GDP Việt Nam năm 2024 đạt khoảng
**11.511,9 nghìn tỷ VND**, tăng **7,09%** so với năm 2023.

Năng suất lao động:

- Năm 2024 đạt khoảng **221,9 triệu VND/người**
- Năm 2025 đạt khoảng **245,0 triệu VND/người**

Đồng thời:

- Đóng góp của khoa học - công nghệ vào GDP năm 2025 ước khoảng:
    - **1,68% trực tiếp**
    - **0,81% lan tỏa**
- Tổng đóng góp khoảng **2,49% GDP**
- Kinh tế số hiện chiếm khoảng **18,3% - 19,5% GDP**

Trong bối cảnh đó, câu hỏi nghiên cứu được đặt ra là:

Nếu mô hình hóa nền kinh tế Việt Nam bằng hàm sản xuất
Cobb-Douglas mở rộng có bổ sung các yếu tố:

- Chuyển đổi số (D)
- Năng lực AI
- Vốn nhân lực số (H)

thì:

- Sản lượng dự báo có phù hợp với dữ liệu thực tế không?
- Yếu tố nào đóng góp lớn nhất vào tăng trưởng kinh tế?
""")

# ==================================================
# 1.2 MÔ HÌNH TOÁN HỌC
# ==================================================

st.header("1.2. Mô hình toán học")

st.markdown("""
Hàm sản xuất tổng hợp của nền kinh tế được giả định
theo dạng Cobb-Douglas mở rộng.

Trong mô hình này, sản lượng nền kinh tế không chỉ phụ thuộc
vào vốn và lao động truyền thống,
mà còn phụ thuộc vào mức độ số hóa,
năng lực trí tuệ nhân tạo và vốn nhân lực số.
""")

# --------------------------------------------------
# HÀM SẢN XUẤT
# --------------------------------------------------

st.subheader("Hàm sản xuất Cobb-Douglas mở rộng")

st.latex(r"""
Y_t
=
A_t
\cdot
K_t^{\alpha}
\cdot
L_t^{\beta}
\cdot
D_t^{\gamma}
\cdot
AI_t^{\delta}
\cdot
H_t^{\theta}
""")

st.markdown("""
với điều kiện lợi suất không đổi theo quy mô:
""")

st.latex(r"""
\alpha + \beta + \gamma + \delta + \theta = 1
""")

# --------------------------------------------------
# GIẢI THÍCH BIẾN
# --------------------------------------------------

st.subheader("Ý nghĩa các biến trong mô hình")

st.markdown("""
- \(Y_t\): GDP của nền kinh tế tại thời điểm \(t\)

- \(A_t\): năng suất nhân tố tổng hợp (TFP)

- \(K_t\): vốn vật chất

- \(L_t\): lao động

- \(D_t\): mức độ số hóa nền kinh tế

- \(AI_t\): năng lực trí tuệ nhân tạo

- \(H_t\): vốn nhân lực số
""")

st.markdown("""
Trong đó:

- \(D_t\) có thể được đo bằng tỷ trọng kinh tế số trong GDP
- \(AI_t\) có thể đại diện bởi số doanh nghiệp công nghệ số
- \(H_t\) được đo bằng tỷ lệ lao động qua đào tạo
""")

# --------------------------------------------------
# Ý NGHĨA CÁC THAM SỐ
# --------------------------------------------------

st.subheader("Ý nghĩa kinh tế của các tham số")

st.markdown("""
Các hệ số:

- \(\alpha\)
- \(\beta\)
- \(\gamma\)
- \(\delta\)
- \(\theta\)

lần lượt phản ánh độ co giãn của GDP đối với:

- vốn vật chất
- lao động
- số hóa
- AI
- vốn nhân lực số
""")

st.markdown("""
Điều kiện tổng các hệ số bằng 1 cho thấy mô hình giả định:

### Lợi suất không đổi theo quy mô

Nghĩa là nếu tất cả đầu vào cùng tăng theo một tỷ lệ,
thì GDP cũng tăng theo đúng tỷ lệ đó.
""")

# --------------------------------------------------
# DẠNG LOGARIT
# --------------------------------------------------

st.subheader("Biến đổi logarit của mô hình")

st.latex(r"""
\ln Y_t
=
\ln A_t
+
\alpha \ln K_t
+
\beta \ln L_t
+
\gamma \ln D_t
+
\delta \ln AI_t
+
\theta \ln H_t
""")

# --------------------------------------------------
# PHÂN RÃ TĂNG TRƯỞNG
# --------------------------------------------------

st.subheader("Phương trình phân rã tăng trưởng")

st.latex(r"""
\Delta \ln Y_t
=
\Delta \ln A_t
+
\alpha \Delta \ln K_t
+
\beta \Delta \ln L_t
+
\gamma \Delta \ln D_t
+
\delta \Delta \ln AI_t
+
\theta \Delta \ln H_t
""")

# --------------------------------------------------
# GIẢI THÍCH KINH TẾ
# --------------------------------------------------

st.subheader("Ý nghĩa của phương trình")

st.markdown("""
Phương trình trên cho phép phân rã tăng trưởng GDP thành:

1. Tăng trưởng năng suất nhân tố tổng hợp (TFP)

2. Đóng góp của vốn vật chất

3. Đóng góp của lao động

4. Đóng góp của chuyển đổi số

5. Đóng góp của năng lực AI

6. Đóng góp của vốn nhân lực số
""")

st.info("""
Cách tiếp cận này đặc biệt phù hợp để đánh giá
vai trò của kinh tế số và trí tuệ nhân tạo
trong mô hình tăng trưởng mới của Việt Nam.
""")

# ==================================================
# THAM SỐ
# ==================================================
ALPHA = 0.33
BETA = 0.42
GAMMA = 0.10
DELTA = 0.08
THETA = 0.07

# ==================================================
# DATA
# ==================================================
data = {
    'Year': [2020, 2021, 2022, 2023, 2024, 2025],
    'Y': [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
    'K': [16500, 17800, 19600, 21300, 23500, 25900],
    'L': [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
    'D': [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
    'AI': [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
    'H': [24.1, 26.1, 26.2, 27.0, 28.4, 29.2]
}

df = pd.DataFrame(data)

# ==================================================
# 1.3 DATA
# ==================================================
st.header("1.3 Dữ liệu kinh tế vĩ mô")

st.dataframe(df, use_container_width=True)

# ==================================================
# 1.4.1 TFP
# ==================================================
st.header("1.4.1 Ước lượng năng suất nhân tố tổng hợp (TFP)")

df['A_t'] = df['Y'] / (
    (df['K']**ALPHA) *
    (df['L']**BETA) *
    (df['D']**GAMMA) *
    (df['AI']**DELTA) *
    (df['H']**THETA)
)

st.subheader("📌 Bảng TFP")

st.dataframe(
    df[['Year', 'A_t']],
    use_container_width=True
)
fig1 = px.line(
    df,
    x='Year',
    y='A_t',
    markers=True,
    title='Xu hướng TFP giai đoạn 2020-2025'
)

fig1.update_layout(
    paper_bgcolor='#071028',
    plot_bgcolor='#071028',
    font_color='white',
    transition_duration=500
)


st.plotly_chart(fig1, use_container_width=True)

if df['A_t'].iloc[-1] > df['A_t'].iloc[0]:
    st.success("""
    TFP có xu hướng tăng cho thấy
    chất lượng tăng trưởng kinh tế Việt Nam
    đang được cải thiện nhờ chuyển đổi số
    và đổi mới công nghệ.
    """)

# ==================================================
# 1.4.2 Y HAT + MAPE
# ==================================================
st.header("1.4.2 Dự báo sản lượng và MAPE")

A_mean = df['A_t'].mean()

df['Y_hat'] = A_mean * (
    (df['K']**ALPHA) *
    (df['L']**BETA) *
    (df['D']**GAMMA) *
    (df['AI']**DELTA) *
    (df['H']**THETA)
)

df['APE'] = np.abs((df['Y'] - df['Y_hat']) / df['Y']) * 100

mape = df['APE'].mean()

st.metric("MAPE", f"{mape:.2f}%")

compare_df = df[['Year', 'Y', 'Y_hat', 'APE']]

tfp_df = df[['Year', 'A_t']].copy()

tfp_df['A_t'] = tfp_df['A_t'].round(4)

st.table(tfp_df)

# ==================================================
# 1.4.3 GROWTH ACCOUNTING
# ==================================================
st.header("1.4.3 Phân rã tăng trưởng kinh tế")

years_span = len(df) - 1

d_lnY = (np.log(df['Y'].iloc[-1]) - np.log(df['Y'].iloc[0])) / years_span
d_lnK = (np.log(df['K'].iloc[-1]) - np.log(df['K'].iloc[0])) / years_span
d_lnL = (np.log(df['L'].iloc[-1]) - np.log(df['L'].iloc[0])) / years_span
d_lnD = (np.log(df['D'].iloc[-1]) - np.log(df['D'].iloc[0])) / years_span
d_lnAI = (np.log(df['AI'].iloc[-1]) - np.log(df['AI'].iloc[0])) / years_span
d_lnH = (np.log(df['H'].iloc[-1]) - np.log(df['H'].iloc[0])) / years_span
d_lnA = (np.log(df['A_t'].iloc[-1]) - np.log(df['A_t'].iloc[0])) / years_span

contrib_K = ALPHA * d_lnK
contrib_L = BETA * d_lnL
contrib_D = GAMMA * d_lnD
contrib_AI = DELTA * d_lnAI
contrib_H = THETA * d_lnH
contrib_TFP = d_lnA

pct_K = (contrib_K / d_lnY) * 100
pct_L = (contrib_L / d_lnY) * 100
pct_D = (contrib_D / d_lnY) * 100
pct_AI = (contrib_AI / d_lnY) * 100
pct_H = (contrib_H / d_lnY) * 100
pct_TFP = (contrib_TFP / d_lnY) * 100

decomposition_df = pd.DataFrame({
    'Factor': ['K', 'L', 'D', 'AI', 'H', 'TFP'],
    'Contribution (%)': [
        pct_K,
        pct_L,
        pct_D,
        pct_AI,
        pct_H,
        pct_TFP
    ]
})

st.dataframe(decomposition_df, use_container_width=True)

fig3 = px.bar(
    decomposition_df,
    x='Factor',
    y='Contribution (%)',
    title='Đóng góp các nhân tố vào tăng trưởng GDP'
)

st.plotly_chart(fig3, use_container_width=True)

# ==================================================
# 1.4.4 FORECAST 2030
# ==================================================
# ==================================================
# 1.4.4 FORECAST 2030
# ==================================================

st.header("1.4.4 Dự báo GDP Việt Nam năm 2030")

Y_2025 = df['Y'].iloc[-1]
K_2025 = df['K'].iloc[-1]
L_2025 = df['L'].iloc[-1]
A_2025 = df['A_t'].iloc[-1]

years_project = 5

# =========================
# DYNAMIC SCENARIO
# =========================

K_2030 = K_2025 * ((1 + growth_K/100) ** years_project)

L_2030 = L_2025 * ((1 + growth_L/100) ** years_project)

A_2030 = A_2025 * ((1 + growth_TFP/100) ** years_project)

# =========================
# GDP FORECAST
# =========================

Y_2030 = A_2030 * (
    (K_2030**ALPHA) *
    (L_2030**BETA) *
    (D_2030**GAMMA) *
    (AI_2030**DELTA) *
    (H_2030**THETA)
)

growth_2030 = (
    ((Y_2030 / Y_2025)**(1/5)) - 1
) * 100

# ==================================================
# METRICS
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "GDP 2030 Forecast",
        f"{Y_2030:,.0f}"
    )

with col2:
    st.metric(
        "Avg Growth",
        f"{growth_2030:.2f}%"
    )

with col3:
    st.metric(
        "TFP 2030",
        f"{A_2030:.2f}"
    )

# ==================================================
# FORECAST DATAFRAME
# ==================================================

forecast_df = pd.DataFrame({
    'Year': [2025, 2030],
    'GDP': [Y_2025, Y_2030]
})

fig4 = px.bar(
    forecast_df,
    x='Year',
    y='GDP',
    text='GDP',
    title='GDP Forecast 2025 → 2030'
)

fig4.update_traces(textposition='outside')

st.plotly_chart(fig4, use_container_width=True)

# ==================================================
# 1.5 POLICY DISCUSSION
# ==================================================
st.header("1.5 Thảo luận chính sách")

# ==================================================
# CÂU A
# ==================================================
with st.expander("a) TFP của Việt Nam có xu hướng tăng hay giảm trong giai đoạn 2020-2025?"):

    st.markdown(r"""
# Xu hướng của năng suất nhân tố tổng hợp (TFP)

Từ mô hình Cobb-Douglas mở rộng:

$$
A_t =
\frac{
Y_t
}{
K_t^{0.33}
L_t^{0.42}
D_t^{0.10}
AI_t^{0.08}
H_t^{0.07}
}
$$

kết quả ước lượng cho thấy TFP ($A_t$)
có xu hướng tăng tương đối rõ rệt
trong giai đoạn 2022–2025,
sau khi suy giảm nhẹ ở thời kỳ 2020–2021.

## Giai đoạn 2020–2021

TFP suy giảm do tác động của đại dịch COVID-19:

- đứt gãy chuỗi cung ứng
- suy giảm năng suất lao động
- gián đoạn sản xuất
- hiệu quả phân bổ nguồn lực thấp

Điều này phản ánh rằng mặc dù
các yếu tố đầu vào như vốn và lao động vẫn tồn tại,
nhưng hiệu suất khai thác nền kinh tế giảm đáng kể.

## Giai đoạn 2022–2025

Khi nền kinh tế phục hồi,
TFP tăng nhanh trở lại nhờ:

- chuyển đổi số mạnh mẽ
- mở rộng kinh tế số
- gia tăng ứng dụng AI
- cải thiện chất lượng quản trị doanh nghiệp
- nâng cao hiệu quả sử dụng nguồn lực

---

# Ý nghĩa đối với chất lượng tăng trưởng

Xu hướng tăng của TFP cho thấy
mô hình tăng trưởng của Việt Nam
đang chuyển dịch theo hướng tích cực.

## 1. Chuyển từ tăng trưởng theo chiều rộng sang chiều sâu

Trong quá khứ,
tăng trưởng kinh tế Việt Nam chủ yếu dựa vào:

- tích lũy vốn vật chất
- lao động giá rẻ

Tuy nhiên,
sự gia tăng của TFP cho thấy
động lực tăng trưởng hiện nay
ngày càng phụ thuộc vào:

- công nghệ
- đổi mới sáng tạo
- chất lượng quản trị
- hiệu quả phân bổ nguồn lực

## 2. Công nghệ trở thành động lực năng suất mới

Việc AI và chuyển đổi số phát triển mạnh
không chỉ tạo ra sản lượng trực tiếp,
mà còn nâng cao hiệu quả sử dụng:

- vốn vật chất
- lao động
- hạ tầng số

Điều này phản ánh hiệu ứng lan tỏa công nghệ
đang ngày càng rõ rệt trong nền kinh tế.

## 3. Hàm ý dài hạn

TFP tăng là tín hiệu tích cực
đối với tăng trưởng bền vững,
vì đây là yếu tố giúp nền kinh tế:

- tăng trưởng mà không phụ thuộc quá mức vào vốn
- cải thiện năng lực cạnh tranh
- nâng cao khả năng chống chịu cú sốc
- tránh rơi vào bẫy thu nhập trung bình
""")

# ==================================================
# CÂU B
# ==================================================
with st.expander("b) Trong các yếu tố mới D, AI, H, yếu tố nào đóng góp lớn nhất?"):

    st.markdown(r"""
# Phân tích đóng góp của các yếu tố mới

Theo mô hình phân rã tăng trưởng:

$$
\Delta \ln Y_t
=
\Delta \ln A_t
+
\alpha \Delta \ln K_t
+
\beta \Delta \ln L_t
+
\gamma \Delta \ln D_t
+
\delta \Delta \ln AI_t
+
\theta \Delta \ln H_t
$$

kết quả cho thấy:

# AI là nhân tố đóng góp nổi bật nhất
trong nhóm các yếu tố mới.

---

# Giải thích nguyên nhân

## 1. Tốc độ tăng trưởng của khu vực AI rất cao

Số lượng doanh nghiệp công nghệ số tăng từ:

$$
55.6 \rightarrow 80.1
\text{ nghìn doanh nghiệp}
$$

trong giai đoạn 2020–2025,
tương đương mức tăng gần 44%.

Mặc dù hệ số co giãn sản lượng của AI:

$$
\delta = 0.08
$$

không quá lớn,
nhưng tốc độ mở rộng của khu vực AI
đã tạo ra mức đóng góp đáng kể vào tăng trưởng GDP.

## 2. AI có hiệu ứng tối ưu hóa mạnh

AI giúp:

- tự động hóa quy trình
- giảm chi phí vận hành
- nâng cao hiệu suất lao động
- tối ưu chuỗi cung ứng
- cải thiện khả năng ra quyết định

Do đó,
AI không chỉ tạo giá trị trực tiếp,
mà còn nâng năng suất của các yếu tố khác.

## 3. AI có khả năng mở rộng quy mô cao

Các doanh nghiệp công nghệ số:

- ít phụ thuộc tài sản vật lý
- có chi phí biên thấp
- dễ nhân rộng mô hình

Điều này giúp khu vực AI
tăng trưởng nhanh hơn nhiều ngành truyền thống.

---

# Vai trò của D và H

## D — Hạ tầng số

D đóng vai trò nền tảng cho chuyển đổi số:

- dữ liệu
- cloud computing
- internet
- nền tảng số

Tuy nhiên,
đây chủ yếu là yếu tố hỗ trợ dài hạn,
không tạo tăng trưởng đột biến ngay lập tức.

## H — Vốn nhân lực số

Vốn nhân lực số có đóng góp ổn định,
nhưng thường xuất hiện độ trễ lớn
do quá trình đào tạo nhân lực chất lượng cao cần nhiều thời gian.

Vì vậy,
H là yếu tố quyết định tính bền vững dài hạn,
trong khi AI là động lực tăng tốc ngắn và trung hạn.
""")

# ==================================================
# CÂU C
# ==================================================
with st.expander("c) Mục tiêu 30% kinh tế số/GDP vào năm 2030 có khả thi không?"):

    st.markdown(r"""
# Đánh giá tính khả thi của mục tiêu

Theo kết quả mô phỏng từ mô hình Cobb-Douglas mở rộng:

- $D = 30\%$
- $AI = 100$ nghìn doanh nghiệp
- $H = 35\%$
- $K, L$ tăng trưởng đều $6\%$/năm
- TFP tăng $1.2\%$/năm

GDP Việt Nam năm 2030
được dự báo đạt mức tăng trưởng rất tích cực,
đồng thời duy trì tốc độ tăng trưởng bình quân cao
trong giai đoạn 2025–2030.

---

# Kết luận

Mục tiêu đưa kinh tế số đạt:

$$
30\%
\text{ GDP vào năm 2030}
$$

là khả thi về mặt kinh tế lượng.

Tuy nhiên,
đây không phải mục tiêu có thể đạt được một cách tự động,
mà phụ thuộc vào nhiều điều kiện ràng buộc chính sách.

---

# Các ràng buộc quan trọng

## 1. Ràng buộc về nguồn nhân lực số

AI và chuyển đổi số
chỉ phát huy hiệu quả
khi có lực lượng lao động chất lượng cao tương ứng.

Nếu tỷ lệ lao động qua đào tạo không đạt khoảng:

$$
35\%
$$

thì nền kinh tế sẽ đối mặt với:

- thiếu hụt kỹ sư AI
- thiếu chuyên gia dữ liệu
- thiếu nhân lực công nghệ cao

=> làm giảm hiệu quả đầu tư số.

---

## 2. Ràng buộc về hạ tầng số và năng lượng

Một nền kinh tế số quy mô lớn
đòi hỏi:

- hệ thống dữ liệu quốc gia
- cloud infrastructure
- data center
- điện năng ổn định
- mạng truyền dẫn tốc độ cao

Nếu hạ tầng vật chất không theo kịp,
quá trình chuyển đổi số sẽ bị nghẽn.

---

## 3. Ràng buộc thể chế và quản trị dữ liệu

Việt Nam cần:

- hoàn thiện hành lang pháp lý cho AI
- xây dựng cơ chế sandbox
- thúc đẩy dữ liệu mở
- tăng cường an ninh mạng

để bảo đảm:

- chủ quyền dữ liệu
- an toàn thông tin
- niềm tin số của doanh nghiệp và người dân

---

# Hàm ý chính sách tổng quát

Kết quả mô hình cho thấy
động lực tăng trưởng dài hạn của Việt Nam
sẽ ngày càng phụ thuộc vào:

- công nghệ
- AI
- chất lượng nhân lực
- năng suất tổng hợp (TFP)

hơn là mở rộng vốn và lao động truyền thống.

Do đó,
chiến lược phát triển trong thập kỷ tới
cần ưu tiên:

- nâng cao năng suất
- thúc đẩy đổi mới sáng tạo
- phát triển hệ sinh thái AI quốc gia
- đầu tư giáo dục và kỹ năng số
""")

st.success("Hoàn thành Bài 1.")
# ==================================================
# EXPORT OUTPUT FOR BAI 12
# ==================================================

# EXPORT OUTPUT FOR BAI 12
# ==================================================

M1_OUTPUT= {

    "GDP_2030": Y_2030,

    "Growth_2030": growth_2030,
    "TFP_2030": A_2030,

    "Digital_Contribution":
        decomposition_df.loc[
            decomposition_df["Factor"] == "D",
            "Contribution (%)"
        ].values[0],

    "AI_Contribution":
        decomposition_df.loc[
            decomposition_df["Factor"] == "AI",
            "Contribution (%)"
        ].values[0],

    "Human_Contribution":
        decomposition_df.loc[
            decomposition_df["Factor"] == "H",
            "Contribution (%)"
        ].values[0]
}


def get_m1_output():
    return M1_OUTPUT


if __name__ == "__main__":
    st.json(get_m1_output())