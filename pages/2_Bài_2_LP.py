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

from scipy.optimize import linprog
from pulp import *

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Bài 2 - Linear Programming",
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
st.title("📘 BÀI 2 - TỐI ƯU HÓA PHÂN BỔ NGÂN SÁCH BẰNG LINEAR PROGRAMMING")
# =========================================================
# SIDEBAR ĐIỀU CHỈNH THAM SỐ
# =========================================================

st.sidebar.header("⚙️ Điều chỉnh tham số")

st.sidebar.subheader("Ngân sách")

B = st.sidebar.slider(
    "Ngân sách tổng (nghìn tỷ VND)",
    min_value=50,
    max_value=300,
    value=100,
    step=10
)

st.sidebar.subheader("Ràng buộc tối thiểu")

min_x1 = st.sidebar.number_input(
    "Hạ tầng số tối thiểu x1",
    value=25.0
)

min_x2 = st.sidebar.number_input(
    "AI tối thiểu x2",
    value=15.0
)

min_x3 = st.sidebar.number_input(
    "Nhân lực số tối thiểu x3",
    value=20.0
)

min_x4 = st.sidebar.number_input(
    "R&D tối thiểu x4",
    value=10.0
)

st.sidebar.subheader("Tỷ trọng công nghệ chiến lược")

tech_ratio = st.sidebar.slider(
    "AI + R&D tối thiểu (%)",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.05
)

st.sidebar.subheader("Hệ số tác động GDP")

coef1 = st.sidebar.number_input(
    "Hạ tầng số",
    value=0.85,
    step=0.05
)

coef2 = st.sidebar.number_input(
    "AI",
    value=1.20,
    step=0.05
)

coef3 = st.sidebar.number_input(
    "Nhân lực số",
    value=0.95,
    step=0.05
)

coef4 = st.sidebar.number_input(
    "R&D",
    value=1.35,
    step=0.05
)

st.sidebar.subheader("Kịch bản nhân lực")

priority_human = st.sidebar.number_input(
    "Ngưỡng x3 ưu tiên",
    value=30.0
)
# =========================================================
# 2.1 BỐI CẢNH
# =========================================================
st.header("2.1. Bối cảnh Việt Nam")

st.markdown("""
Theo Quyết định số 749/QĐ-TTg về Chương trình Chuyển đổi số quốc gia,
Việt Nam đặt mục tiêu kinh tế số đạt khoảng 20% GDP vào năm 2025.

Giả sử Chính phủ có 100 nghìn tỷ VND ngân sách đầu tư số năm 2026,
phân bổ cho 4 hạng mục:

- Hạ tầng số
- AI và dữ liệu
- Nhân lực số
- R&D công nghệ

Mục tiêu là tối đa hóa mức tăng GDP kỳ vọng.
""")

# =========================================================
# 2.2 MÔ HÌNH TOÁN HỌC
# =========================================================
st.header("2.2. Mô hình toán học")

st.latex(r'''
\max Z
=
0.85x_1
+
1.20x_2
+
0.95x_3
+
1.35x_4
''')

st.markdown("""
Trong đó:

- \(x_1\): Đầu tư hạ tầng số
- \(x_2\): Đầu tư AI và dữ liệu
- \(x_3\): Đầu tư nhân lực số
- \(x_4\): Đầu tư R&D công nghệ
""")

st.subheader("Ràng buộc")

st.latex(r'''
x_1 + x_2 + x_3 + x_4 \leq 100
''')

st.latex(r'''
x_1 \geq 25
''')

st.latex(r'''
x_2 \geq 15
''')

st.latex(r'''
x_3 \geq 20
''')

st.latex(r'''
x_4 \geq 10
''')

st.latex(r'''
x_2 + x_4 \geq 0.35(x_1+x_2+x_3+x_4)
''')

# =========================================================
# 2.3 DIỄN GIẢI HỆ SỐ
# =========================================================
st.header("2.3. Diễn giải hệ số mục tiêu")

coef_df = pd.DataFrame({
    "Hạng mục": [
        "Hạ tầng số",
        "AI và dữ liệu",
        "Nhân lực số",
        "R&D công nghệ"
    ],
    "Hệ số tác động GDP": [
        coef1,
        coef2,
        coef3,
        coef4
    ]
})

st.dataframe(coef_df, use_container_width=True)

st.markdown("""
- R&D có hệ số cao nhất do tác động lan tỏa công nghệ dài hạn.
- AI tạo tăng trưởng nhanh nhờ khả năng tối ưu hóa dữ liệu và tự động hóa.
- Nhân lực số đóng vai trò nền tảng hấp thụ công nghệ.
- Hạ tầng số là điều kiện cần cho toàn bộ quá trình chuyển đổi số.
""")

# =========================================================
# 2.4.1 SCIPY LINPROG
# =========================================================
st.header("2.4.1. Giải bài toán bằng scipy.optimize.linprog")

# =========================================================
# HÀM MỤC TIÊU
# =========================================================
c = [
    -coef1,
    -coef2,
    -coef3,
    -coef4
]

# =========================================================
# RÀNG BUỘC
# =========================================================
A_ub = [
    [1,1,1,1],
    [-1,0,0,0],
    [0,-1,0,0],
    [0,0,-1,0],
    [0,0,0,-1],
    [
        tech_ratio,
        -(1-tech_ratio),
        tech_ratio,
        -(1-tech_ratio)
    ]
]
b_ub = [
    B,
    -min_x1,
    -min_x2,
    -min_x3,
    -min_x4,
    0
]


bounds = [(0, None)] * 4

# =========================================================
# SOLVE
# =========================================================
res = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=bounds,
    method='highs'
)

x1, x2, x3, x4 = res.x

Z_optimal = -res.fun

# =========================================================
# KẾT QUẢ
# =========================================================
result_df = pd.DataFrame({
    "Hạng mục": [
        "Hạ tầng số (x1)",
        "AI và dữ liệu (x2)",
        "Nhân lực số (x3)",
        "R&D công nghệ (x4)"
    ],
    "Giá trị tối ưu": [
        x1,
        x2,
        x3,
        x4
    ]
})

st.subheader("Phân bổ ngân sách tối ưu")

st.dataframe(
    result_df.style.format({
        "Giá trị tối ưu": "{:.2f}"
    }),
    use_container_width=True
)

st.success(
    f"Giá trị GDP kỳ vọng tối đa: {Z_optimal:.2f} nghìn tỷ VND"
)

# =========================================================
# PIE CHART
# =========================================================
fig_pie = px.pie(
    result_df,
    names="Hạng mục",
    values="Giá trị tối ưu",
    title="Cơ cấu phân bổ ngân sách tối ưu"
)

st.plotly_chart(fig_pie, use_container_width=True)

# =========================================================
# INTERPRETATION
# =========================================================
st.subheader("Phân tích kết quả tối ưu")

st.markdown(f"""
Kết quả tối ưu cho thấy:

- Hạ tầng số được giữ ở mức tối thiểu:
  
\[
x_1 = {x1:.0f}
\]

- AI và dữ liệu được duy trì ở mức:

\[
x_2 = {x2:.0f}
\]

- Nhân lực số được duy trì ở mức:

\[
x_3 = {x3:.0f}
\]

- Phần lớn ngân sách còn lại được phân bổ cho R&D:

\[
x_4 = {x4:.0f}
\]

Nguyên nhân là vì R&D có hệ số tác động GDP lớn nhất:

\[
1.35
\]

cao hơn AI (1.20),
nhân lực số (0.95)
và hạ tầng số (0.85).

Do đó,
mô hình tối ưu sẽ tự động dồn nguồn lực vào lĩnh vực
có hiệu quả sinh GDP cận biên cao nhất.

Điều này phản ánh logic tối ưu hóa kinh tế:
nguồn vốn khan hiếm sẽ được ưu tiên cho khu vực
mang lại lợi ích kinh tế lớn nhất.
""")

# =========================================================
# 2.4.2 PULP + DUAL VALUES
# =========================================================
st.header("2.4.2. Giải bằng PuLP và phân tích giá đối ngẫu")

prob = LpProblem("Digital_Investment", LpMaximize)

# VARIABLES
x1_p = LpVariable("x1", lowBound=0)
x2_p = LpVariable("x2", lowBound=0)
x3_p = LpVariable("x3", lowBound=0)
x4_p = LpVariable("x4", lowBound=0)

# OBJECTIVE
prob += (
    0.85*x1_p
    + 1.20*x2_p
    + 0.95*x3_p
    + 1.35*x4_p
)

# CONSTRAINTS
prob += x1_p + x2_p + x3_p + x4_p <= 100, "Budget"
prob += x1_p >= 25, "Infrastructure"
prob += x2_p >= 15, "AI"
prob += x3_p >= 20, "Human"
prob += x4_p >= 10, "RD"

prob += (
    x2_p + x4_p
    >=
    0.35*(x1_p+x2_p+x3_p+x4_p)
), "Strategic"

prob.solve()

# =========================================================
# DUAL VALUES
# =========================================================
dual_rows = []

for name, constraint in prob.constraints.items():

    dual_rows.append({
        "Ràng buộc": name,
        "Shadow Price": constraint.pi
    })

dual_df = pd.DataFrame(dual_rows)

st.subheader("Giá đối ngẫu của các ràng buộc")

st.dataframe(
    dual_df.style.format({
        "Shadow Price": "{:.4f}"
    }),
    use_container_width=True
)

# =========================================================
# POLICY INTERPRETATION
# =========================================================
st.subheader("Ý nghĩa kinh tế của shadow price")

st.markdown("""
Shadow price của ràng buộc ngân sách tổng
phản ánh mức GDP kỳ vọng tăng thêm
nếu Chính phủ tăng thêm 1 đơn vị ngân sách.

Đây là:

- Giá trị cận biên của vốn công
- Hiệu quả đầu tư bổ sung
- Mức độ khan hiếm của nguồn lực ngân sách

Nếu shadow price lớn:

- Nền kinh tế vẫn còn thiếu vốn đầu tư số
- Đầu tư công nghệ vẫn mang lại hiệu quả cao
- Việt Nam chưa đạt trạng thái bão hòa đầu tư số

Tuy nhiên,
shadow price chỉ phản ánh hiệu quả trong vùng lân cận nghiệm tối ưu.

Trong dài hạn,
khi ngân sách tiếp tục mở rộng,
hiệu quả đầu tư cận biên có thể giảm dần do:

- Năng lực hấp thụ công nghệ hạn chế
- Thiếu nhân lực chất lượng cao
- Hiệu suất đầu tư giảm theo quy mô
""")

# =========================================================
# 2.4.3 PHÂN TÍCH ĐỘ NHẠY
# ========# =========================================================
# 2.4.3 PHÂN TÍCH ĐỘ NHẠY
# =========================================================
st.header("2.4.3. Phân tích độ nhạy ngân sách")

budgets = [100, 120, 140]
optimal_values = []

for budget in budgets:

    b_test = [
        budget,
        -min_x1,
        -min_x2,
        -min_x3,
        -min_x4,
        0
    ]

    res_test = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_test,
        bounds=bounds,
        method="highs"
    )

    if res_test.success:
        optimal_values.append(-res_test.fun)
    else:
        optimal_values.append(np.nan)
        st.warning(
            f"Ngân sách {budget} không khả thi: {res_test.message}"
        )

sensitivity_df = pd.DataFrame({
    "Ngân sách": budgets,
    "GDP tối ưu": optimal_values
})

st.dataframe(sensitivity_df, use_container_width=True)

fig_line = px.line(
    sensitivity_df,
    x="Ngân sách",
    y="GDP tối ưu",
    markers=True,
    title="Đường cong Z*(B)"
)

st.plotly_chart(fig_line, use_container_width=True)

# =========================================================
# INTERPRETATION
# =========================================================
st.subheader("Nhận xét phân tích độ nhạy")

st.markdown("""
Khi ngân sách tăng từ:

- 100 → 120
- 120 → 140

GDP tối ưu tiếp tục tăng mạnh.

Điều này cho thấy:

- Việt Nam vẫn còn dư địa đầu tư số lớn
- Đầu tư công nghệ chưa đạt trạng thái bão hòa
- Mỗi đồng vốn bổ sung vẫn tạo thêm tăng trưởng đáng kể

Tuy nhiên,
nếu ngân sách tăng quá nhanh
mà không đi kèm cải thiện:

- Nhân lực số
- Thể chế quản lý
- Năng lực hấp thụ công nghệ

thì hiệu quả đầu tư cận biên có thể suy giảm.
""")

# =========================================================
# 2.4.4 KỊCH BẢN ƯU TIÊN NHÂN LỰC SỐ
# =========================================================
st.header("2.4.4. Kịch bản ưu tiên nhân lực số")

b_ub_new = [
    B,
    -min_x1,
    -min_x2,
    -priority_human,
    -min_x4,
    0
]

res_new = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_ub_new,
    bounds=bounds,
    method='highs'
)

if res_new.success:

    x1_new, x2_new, x3_new, x4_new = res_new.x

    Z_new = -res_new.fun

    st.success("Bài toán vẫn khả thi.")

    scenario_df = pd.DataFrame({
        "Hạng mục": [
            "Hạ tầng số",
            "AI",
            "Nhân lực số",
            "R&D"
        ],
        "Phân bổ mới": [
            x1_new,
            x2_new,
            x3_new,
            x4_new
        ]
    })

    st.dataframe(
        scenario_df.style.format({
            "Phân bổ mới": "{:.2f}"
        }),
        use_container_width=True
    )

    st.metric(
        "GDP tối ưu mới",
        f"{Z_new:.2f}"
    )

else:

    st.error("Bài toán không khả thi.")

# =========================================================
# INTERPRETATION
# =========================================================
st.subheader("Phân tích kịch bản ưu tiên nhân lực số")

st.markdown("""
Khi Chính phủ tăng ràng buộc:

\[
x_3 \geq 30
\]

một phần ngân sách phải chuyển từ R&D hoặc AI
sang đào tạo nhân lực số.

Điều này giúp:

- Giảm thiếu hụt kỹ sư AI
- Tăng năng lực hấp thụ công nghệ
- Cải thiện nền tảng đổi mới dài hạn

Tuy nhiên,
GDP tối ưu ngắn hạn sẽ giảm nhẹ
vì nguồn vốn bị rút khỏi các lĩnh vực
có hiệu quả sinh GDP cận biên cao hơn.

Đây là sự đánh đổi giữa:

- Tăng trưởng ngắn hạn
và
- Năng lực công nghệ dài hạn.
""")

# =========================================================
# 2.5 THẢO LUẬN CHÍNH SÁCH
# =========================================================
st.header("2.5. Thảo luận chính sách")

# =========================================================
# CÂU A
# =========================================================
st.subheader(
    "a) Khi ngân sách tổng tăng thêm 1 tỷ VND, GDP kỳ vọng tăng thêm bao nhiêu?"
)

st.markdown("""
Theo lý thuyết quy hoạch tuyến tính,
mức tăng của GDP kỳ vọng khi ngân sách tăng thêm 1 đơn vị
được phản ánh bởi shadow price của ràng buộc ngân sách tổng.

Điều này có nghĩa:

- Nếu shadow price bằng 1.35,
thì về mặt lý thuyết,
1 tỷ VND ngân sách bổ sung
có thể tạo thêm khoảng 1.35 tỷ VND GDP kỳ vọng.

Shadow price phản ánh:

- Giá trị cận biên của vốn công
- Hiệu quả đầu tư bổ sung
- Mức độ khan hiếm của nguồn lực ngân sách

Tuy nhiên,
đây không hoàn toàn là cận trên tuyệt đối
của chi phí cơ hội vốn công.

Nguyên nhân là vì:

- Shadow price chỉ đúng quanh nghiệm tối ưu hiện tại
- Hiệu quả đầu tư có thể giảm dần khi quy mô đầu tư tăng mạnh
- Nền kinh tế có giới hạn hấp thụ công nghệ
- Chất lượng quản trị công ảnh hưởng lớn tới hiệu quả thực tế

Do đó,
shadow price nên được hiểu là:

"Hiệu quả cận biên ngắn hạn của vốn đầu tư công nghệ"
hơn là một giá trị cố định dài hạn.
""")

# =========================================================
# CÂU B
# =========================================================
st.subheader(
    "b) Vì sao R&D có hệ số tác động cao nhất nhưng ràng buộc tối thiểu thấp nhất?"
)

st.markdown("""
R&D có hệ số tác động GDP cao nhất vì:

- Tạo đổi mới công nghệ
- Tăng năng suất dài hạn
- Hình thành công nghệ lõi
- Tạo hiệu ứng lan tỏa toàn nền kinh tế

Tuy nhiên,
R&D cũng là lĩnh vực:

- Rủi ro cao
- Khó đo lường hiệu quả
- Có độ trễ dài
- Phụ thuộc mạnh vào chất lượng nhân lực

Trong ngắn hạn,
đầu tư R&D chưa chắc tạo ra tăng trưởng ngay lập tức.

Ngoài ra,
năng lực hấp thụ R&D của Việt Nam hiện nay
vẫn còn hạn chế do:

- Thiếu chuyên gia công nghệ cao
- Hệ sinh thái đổi mới sáng tạo chưa hoàn thiện
- Liên kết giữa đại học và doanh nghiệp còn yếu

Vì vậy,
Chính phủ chỉ đặt mức tối thiểu thấp
để duy trì tính linh hoạt ngân sách
và tránh tạo áp lực giải ngân quá lớn
vào khu vực có độ rủi ro cao.
""")

# =========================================================
# CÂU C
# =========================================================
st.subheader(
    "c) Tỷ lệ 35% công nghệ chiến lược có khả thi không?"
)

st.markdown("""
Mục tiêu duy trì tối thiểu 35% ngân sách
cho AI và R&D là mục tiêu khá tham vọng
trong bối cảnh Việt Nam hiện nay.

Nguyên nhân là vì ngân sách Nhà nước
vẫn phải ưu tiên cho nhiều lĩnh vực khác như:

- Hạ tầng giao thông
- Y tế
- Giáo dục
- An sinh xã hội
- Quốc phòng

Ngoài ra,
AI và R&D đòi hỏi:

- Nhân lực chất lượng cao
- Hạ tầng dữ liệu mạnh
- Năng lực nghiên cứu lớn
- Hệ sinh thái đổi mới sáng tạo hoàn chỉnh

Trong khi đó,
khả năng hấp thụ công nghệ của nhiều doanh nghiệp Việt Nam
vẫn còn hạn chế.

Tuy nhiên,
về dài hạn,
mục tiêu này vẫn khả thi nếu Việt Nam:

- Thúc đẩy hợp tác công tư (PPP)
- Khuyến khích doanh nghiệp tư nhân đầu tư R&D
- Thu hút FDI công nghệ cao
- Hoàn thiện hành lang pháp lý cho AI
- Phát triển giáo dục và kỹ năng số
- Xây dựng hạ tầng dữ liệu quốc gia

Nếu thực hiện tốt các điều kiện trên,
tỷ trọng đầu tư công nghệ chiến lược cao
có thể trở thành động lực quan trọng
giúp Việt Nam chuyển đổi sang mô hình tăng trưởng dựa trên năng suất và đổi mới sáng tạo.
""")

st.success("Hoàn thành toàn bộ Bài 2.")