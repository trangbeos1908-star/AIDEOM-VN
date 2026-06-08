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
import pulp
import matplotlib.pyplot as plt

# ==================================================
# CẤU HÌNH TRANG
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
st.set_page_config(
    page_title="Bài 5 - Lựa chọn dự án chuyển đổi số",
    layout="wide"
)
st.title("📊 Bài 5- Quy hoạch nguyên hỗn hợp (MIP) lựa chọn dự án chuyển đổi số")
# ==================================================
# SIDEBAR ĐIỀU CHỈNH THAM SỐ
# ==================================================
st.sidebar.header("⚙️ Điều chỉnh mô hình bài 5")

budget_total = st.sidebar.slider(
    "Ngân sách tổng 5 năm",
    50000, 120000, 80000, 5000
)

budget_year12 = st.sidebar.slider(
    "Ngân sách năm 1-2",
    20000, 70000, 40000, 5000
)

min_projects = st.sidebar.slider(
    "Số dự án tối thiểu",
    1, 15, 7, 1
)

max_projects = st.sidebar.slider(
    "Số dự án tối đa",
    1, 15, 11, 1
)

use_exclude_datacenter = st.sidebar.checkbox(
    "Chỉ chọn một trong P1 và P2",
    value=True
)

use_prerequisite = st.sidebar.checkbox(
    "Bật ràng buộc tiên quyết P8/P13 cần P12",
    value=True
)

require_gov_project = st.sidebar.checkbox(
    "Bắt buộc ít nhất một Chính phủ số P4/P5",
    value=True
)

require_cybersecurity = st.sidebar.checkbox(
    "Bắt buộc P14 An ninh mạng",
    value=True
)

force_p1_p2 = st.sidebar.checkbox(
    "Kịch bản bắt buộc chọn cả P1 và P2",
    value=False
)

use_risk_objective = st.sidebar.checkbox(
    "Tối đa hóa lợi ích kỳ vọng có xét rủi ro",
    value=False
)

relaxed_budget = st.sidebar.slider(
    "Ngân sách kịch bản nới rộng",
    80000, 150000, 100000, 5000
)

if min_projects > max_projects:
    st.error("Số dự án tối thiểu không được lớn hơn số dự án tối đa.")
    st.stop()

if force_p1_p2 and use_exclude_datacenter:
    st.warning("Đang bật cả 'chỉ chọn một P1/P2' và 'bắt buộc chọn cả P1/P2'. Mô hình có thể vô nghiệm.")
# ==================================================
# 5.4. YÊU CẦU LẬP TRÌNH
# ==================================================



import pandas as pd
import numpy as np
from pulp import *

# ==================================================
# DỮ LIỆU DỰ ÁN
# ==================================================

P = list(range(1,16))

project_names = {
    1:"Trung tâm dữ liệu quốc gia Hòa Lạc",
    2:"Trung tâm dữ liệu quốc gia phía Nam",
    3:"Hệ thống 5G phủ sóng toàn quốc",
    4:"Hệ thống định danh điện tử VNeID 2.0",
    5:"Cổng dịch vụ công quốc gia v3",
    6:"Y tế số quốc gia",
    7:"Giáo dục số K-12 toàn quốc",
    8:"Trung tâm AI quốc gia + supercomputing",
    9:"Sandbox tài chính số",
    10:"Logistics thông minh + cảng biển số",
    11:"Nông nghiệp số ĐBSCL",
    12:"Đào tạo 50.000 kỹ sư AI/bán dẫn",
    13:"Khu CN bán dẫn Bắc Ninh - Bắc Giang",
    14:"An ninh mạng quốc gia",
    15:"Open Data + dữ liệu mở quốc gia"
}

fields = {
    1:"Hạ tầng",
    2:"Hạ tầng",
    3:"Hạ tầng",
    4:"Chính phủ số",
    5:"Chính phủ số",
    6:"Y tế số",
    7:"Giáo dục",
    8:"AI",
    9:"Tài chính số",
    10:"Logistics",
    11:"Nông nghiệp",
    12:"Nhân lực",
    13:"Bán dẫn",
    14:"An ninh",
    15:"Dữ liệu"
}

# Chi phí tổng
C = {
    1:12000,2:11500,3:18000,4:4500,5:3200,
    6:5800,7:6500,8:15000,9:2500,10:7200,
    11:4800,12:8500,13:20000,14:3800,15:1500
}

# Chi phí năm 1-2
C1 = {
    1:8500,2:7500,3:12000,4:3500,5:2500,
    6:4000,7:4500,8:9000,9:1800,10:5000,
    11:3500,12:5500,13:13000,14:2800,15:1200
}

# Lợi ích NPV
B = {
    1:21500,2:20800,3:32500,4:9200,5:6800,
    6:11400,7:12200,8:28500,9:5800,10:13800,
    11:8500,12:16200,13:35000,14:7500,15:3800
}
prob = {
    1:0.85, 2:0.85, 3:0.85,
    4:0.75, 5:0.75,
    6:0.80, 7:0.80,
    8:0.65,
    9:0.80, 10:0.80, 11:0.80,
    12:0.80,
    13:0.65,
    14:0.80, 15:0.80
}
def solve_project_model(
    model_name,
    total_budget,
    year12_budget,
    risk_objective=False,
    force_both_data_centers=False
):

    model = LpProblem(model_name, LpMaximize)

    y_var = LpVariable.dicts(
        "y",
        P,
        cat="Binary"
    )

    return model, y_var

st.subheader("5.1. Bối cảnh Việt Nam")

st.markdown("""
Bộ Thông tin và Truyền thông (nay là Bộ Khoa học và Công nghệ sau hợp nhất 2025) đang xem xét 15 dự án ứng cử
cho chương trình chuyển đổi số quốc gia giai đoạn 2026-2030. Tổng ngân sách dành cho chương trình là 80.000 tỷ
VND. Mỗi dự án có chi phí, lợi ích kỳ vọng và một số ràng buộc đặc thù. Sinh viên cần xây dựng mô hình MIP để
chọn tập dự án tối ưu.
""")
st.subheader("5.3. Mô hình toán học")

st.markdown(r"""
### Biến quyết định
- \( y_i \in \{0,1\}, \; i = 1,...,15 \)
- \( y_i = 1 \) nếu chọn dự án \( i \)

### Hàm mục tiêu
""")

st.latex(r"""
\max \sum_i B_i \cdot y_i
""")

st.markdown("""
### Ràng buộc
""")

st.latex(r"""
\sum_i C_i \cdot y_i \leq 80000
""")
st.markdown("(C1) Ngân sách tổng 5 năm")

st.latex(r"""
\sum_i C_{1,i} \cdot y_i \leq 40000
""")
st.markdown("(C2) Ngân sách năm 1-2")

st.latex(r"""
y_1 + y_2 \leq 1
""")
st.markdown("(C3) Chỉ chọn một trung tâm dữ liệu")

st.latex(r"""
y_8 \leq y_{12}
""")
st.markdown("(C4) AI quốc gia cần đào tạo kỹ sư")

st.latex(r"""
y_{13} \leq y_{12}
""")
st.markdown("(C5) Khu bán dẫn cần đào tạo")

st.latex(r"""
y_4 + y_5 \geq 1
""")
st.markdown("(C6a) Ít nhất một dự án chính phủ số")

st.latex(r"""
y_{14} \geq 1
""")
st.markdown("(C6b) An ninh mạng là bắt buộc")

st.latex(r"""
7 \leq \sum_i y_i \leq 11
""")
st.markdown("(C7) Giới hạn số lượng dự án")
# ==================================================
# 5.4.1 — GIẢI BÀI TOÁN GỐC
# ==================================================

st.subheader("5.4.1. Giải bài toán bằng PuLP với CBC Solver")

m = LpProblem('VN_Project_Selection', LpMaximize)

y = LpVariable.dicts('y', P, cat='Binary')

# Hàm mục tiêu
if use_risk_objective:
    m += lpSum(prob[i] * B[i] * y[i] for i in P)
else:
    m += lpSum(B[i] * y[i] for i in P)

# Ràng buộc
m += lpSum(C[i] * y[i] for i in P) <= budget_total
m += lpSum(C1[i] * y[i] for i in P) <= budget_year12
if use_exclude_datacenter:
    m += y[1] + y[2] <= 1

if use_prerequisite:
    m += y[8] <= y[12]
    m += y[13] <= y[12]

if require_gov_project:
    m += y[4] + y[5] >= 1

if require_cybersecurity:
    m += y[14] >= 1

m += lpSum(y[i] for i in P) >= min_projects
m += lpSum(y[i] for i in P) <= max_projects

# Solve
m.solve(PULP_CBC_CMD(msg=False))

selected_projects = []

for i in P:
    if y[i].value() > 0.5:
        selected_projects.append([
            f"P{i}",
            project_names[i],
            fields[i],
            C[i],
            B[i],
            round(B[i] / C[i], 2)
        ])

df_selected = pd.DataFrame(
    selected_projects,
    columns=[
        "Mã",
        "Tên dự án",
        "Lĩnh vực",
        "Chi phí",
        "Lợi ích NPV",
        "NPV biên"
    ]
)

total_cost = sum(C[i] for i in P if y[i].value() > 0.5)
total_benefit = value(m.objective)

st.write("### Danh sách dự án được lựa chọn")

st.dataframe(
    df_selected,
    use_container_width=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Tổng chi phí",
        f"{total_cost:,.0f} tỷ VND"
    )

with col2:
    st.metric(
        "Tổng lợi ích NPV",
        f"{total_benefit:,.0f} tỷ VND"
    )

with col3:
    st.metric(
        "NPV biên",
        f"{(total_benefit/total_cost):.2f}"
    )

st.caption(
    f"Ngân sách tổng = {budget_total:,} tỷ | "
    f"Ngân sách năm 1-2 = {budget_year12:,} tỷ | "
    f"Số dự án được chọn = {len(selected_projects)}"
)

st.markdown("""
### Nhận xét chính sách

Mô hình lựa chọn tập hợp dự án sao cho tối đa hóa tổng lợi ích kinh tế ròng (NPV)
trong khi vẫn đáp ứng các ràng buộc ngân sách, an ninh mạng,
đào tạo nhân lực và cân bằng chiến lược quốc gia.

Các dự án AI, hạ tầng số và bán dẫn thường có lợi ích tuyệt đối rất cao,
tuy nhiên vẫn bị giới hạn bởi các ràng buộc phụ thuộc kỹ thuật và ngân sách giai đoạn đầu.
""")

# ==================================================
# 5.4.2 — NỚI NGÂN SÁCH
# ==================================================

st.subheader("5.4.2. Phân tích khi nới ngân sách lên 100.000 tỷ VND")

m2 = LpProblem('VN_Project_Selection_100K', LpMaximize)

y2 = LpVariable.dicts('y2', P, cat='Binary')

m2 += lpSum(B[i] * y2[i] for i in P)


m2 += lpSum(C[i] * y2[i] for i in P) <= relaxed_budget
m2 += lpSum(C1[i] * y2[i] for i in P) <= 40000

m2 += y2[1] + y2[2] <= 1
m2 += y2[8] <= y2[12]
m2 += y2[13] <= y2[12]
m2 += y2[4] + y2[5] >= 1
m2 += y2[14] >= 1

m2 += lpSum(y2[i] for i in P) >= 7
m2 += lpSum(y2[i] for i in P) <= 11

m2.solve(PULP_CBC_CMD(msg=False))

selected_100k = []

for i in P:
    if y2[i].value() > 0.5:
        selected_100k.append(project_names[i])

st.write("### Các dự án được chọn khi ngân sách tăng lên 100.000 tỷ")

for p in selected_100k:
    st.write(f"- {p}")

st.markdown("""
### Phân tích thay đổi

Khi ngân sách tổng được mở rộng lên 100.000 tỷ VND,
mô hình bắt đầu lựa chọn thêm các dự án có lợi ích dài hạn lớn nhưng chi phí đầu tư cao,
đặc biệt là các dự án AI và bán dẫn.

Điều này phản ánh đặc điểm của đầu tư công nghệ:
nhiều dự án chiến lược có NPV rất lớn nhưng thường bị loại bỏ trong điều kiện ngân sách hạn chế
do chi phí ban đầu quá cao.

Khi Nhà nước nới trần đầu tư,
mô hình ưu tiên mạnh hơn cho các dự án có khả năng tạo bước nhảy năng suất dài hạn.
""")

# ==================================================
# 5.4.3 — BẮT BUỘC P1 VÀ P2
# ==================================================

st.subheader("5.4.3. Trường hợp Quốc hội yêu cầu phải có cả P1 và P2")

m3 = LpProblem('VN_Project_Selection_P1P2', LpMaximize)

y3 = LpVariable.dicts('y3', P, cat='Binary')

m3 += lpSum(B[i] * y3[i] for i in P)

m3 += lpSum(C[i] * y3[i] for i in P) <= 80000
m3 += lpSum(C1[i] * y3[i] for i in P) <= 40000

# Bắt buộc P1 và P2
if force_p1_p2:
    m3 += y3[1] == 1
    m3 += y3[2] == 1

m3 += y3[8] <= y3[12]
m3 += y3[13] <= y3[12]
m3 += y3[4] + y3[5] >= 1
m3 += y3[14] >= 1

m3 += lpSum(y3[i] for i in P) >= 7
m3 += lpSum(y3[i] for i in P) <= 11

status = m3.solve(PULP_CBC_CMD(msg=False))

if LpStatus[status] == "Optimal":

    Z_new = value(m3.objective)

    st.success("Bài toán vẫn khả thi.")

    st.metric(
        "Tổng lợi ích mới Z*",
        f"{Z_new:,.0f} tỷ VND"
    )

else:
    st.error("Bài toán không còn khả thi.")

st.markdown("""
### Phân tích chính sách

Việc yêu cầu đồng thời cả P1 và P2 phản ánh chiến lược redundancy
(dự phòng hạ tầng dữ liệu quốc gia).

Mặc dù điều này giúp tăng độ an toàn hệ thống,
nó đồng thời làm giảm tính linh hoạt tối ưu hóa của mô hình.

Hai trung tâm dữ liệu quốc gia có chi phí đầu tư rất lớn,
vì vậy việc bắt buộc đồng thời cả hai dự án sẽ làm giảm khả năng lựa chọn
các dự án AI, giáo dục số hoặc logistics số khác.

Đây là ví dụ điển hình của đánh đổi giữa:
- hiệu quả kinh tế tối đa
- và an ninh chiến lược quốc gia.
""")

# ==================================================
# 5.4.4 — THÊM RỦI RO DỰ ÁN
# ==================================================

st.subheader("5.4.4. Mô hình mở rộng có xét rủi ro dự án")

prob = {
    1:0.85,
    2:0.85,
    3:0.85,
    4:0.75,
    5:0.75,
    6:0.80,
    7:0.80,
    8:0.65,
    9:0.80,
    10:0.80,
    11:0.80,
    12:0.80,
    13:0.65,
    14:0.80,
    15:0.80
}

m4 = LpProblem('VN_Project_Risk', LpMaximize)

y4 = LpVariable.dicts('y4', P, cat='Binary')

# Hàm mục tiêu kỳ vọng
m4 += lpSum(prob[i] * B[i] * y4[i] for i in P)

# Constraints
m4 += lpSum(C[i] * y4[i] for i in P) <= 80000
m4 += lpSum(C1[i] * y4[i] for i in P) <= 40000

m4 += y4[1] + y4[2] <= 1
m4 += y4[8] <= y4[12]
m4 += y4[13] <= y4[12]
m4 += y4[4] + y4[5] >= 1
m4 += y4[14] >= 1

m4 += lpSum(y4[i] for i in P) >= 7
m4 += lpSum(y4[i] for i in P) <= 11

m4.solve(PULP_CBC_CMD(msg=False))

risk_projects = []

for i in P:
    if y4[i].value() > 0.5:
        risk_projects.append([
            f"P{i}",
            project_names[i],
            prob[i]
        ])

df_risk = pd.DataFrame(
    risk_projects,
    columns=["Mã", "Dự án", "Xác suất hoàn thành"]
)

st.write("### Các dự án được chọn khi xét rủi ro")

st.dataframe(
    df_risk,
    use_container_width=True
)

st.metric(
    "Lợi ích kỳ vọng E[Z]",
    f"{value(m4.objective):,.0f} tỷ VND"
)

st.markdown("""
### Nhận xét

Khi đưa rủi ro thực hiện dự án vào mô hình,
các dự án AI và bán dẫn bắt đầu bị bất lợi tương đối
do xác suất hoàn thành đúng tiến độ thấp hơn.

Điều này phản ánh thực tế rằng:
các dự án công nghệ lõi thường có:
- độ phức tạp kỹ thuật cao
- phụ thuộc nhân lực chất lượng cao
- và rủi ro triển khai lớn hơn hạ tầng truyền thống.

Do đó,
mô hình tối ưu kỳ vọng sẽ ưu tiên hơn cho các dự án có mức lợi ích ổn định,
thay vì chỉ nhìn vào lợi ích tuyệt đối.
""")

# ==================================================
# 5.5 — CÂU HỎI THẢO LUẬN CHÍNH SÁCH
# ==================================================

st.header("5.5. Câu hỏi thảo luận chính sách")

# --------------------------------------------------

st.subheader("""
a) Vì sao mô hình bỏ qua dự án P15 (Open Data) dù tỷ suất lợi ích/chi phí rất cao?.Đây có phải là kết quả mong muốn về mặt chính sách?
""")

st.markdown("""
Mặc dù dự án P15 có tỷ suất lợi ích trên chi phí rất cao,
quy mô tuyệt đối của lợi ích tạo ra vẫn tương đối nhỏ so với các dự án chiến lược quốc gia khác.

Trong điều kiện ngân sách hữu hạn,
mô hình tối ưu tuyến tính sẽ ưu tiên các dự án có khả năng tạo ra tổng lợi ích tuyệt đối lớn hơn,
đặc biệt là:
- Hạ tầng số quy mô quốc gia
- AI
- Bán dẫn
- Logistics số

Điều này cho thấy một hạn chế quan trọng của tối ưu hóa định lượng:
mô hình có thể đánh giá thấp các dự án nền tảng dữ liệu mở,
trong khi trên thực tế Open Data thường tạo ra hiệu ứng lan tỏa rất mạnh cho:
- Đổi mới sáng tạo
- Startup
- Minh bạch hóa quản trị công
- Và phát triển hệ sinh thái AI dài hạn.

Vì vậy,
nếu chỉ dựa vào tối ưu NPV ngắn hạn,
Chính phủ có thể vô tình cắt giảm các khoản đầu tư nền tảng chiến lược.

Đây không hẳn là kết quả mong muốn về mặt chính sách công dài hạn.
""")

# --------------------------------------------------

st.subheader("""
b) Ràng buộc “bắt buộc P14 (an ninh mạng)” có làm giảm Z* không? Việc bắt buộc này có hợp lý không?
""")

st.markdown("""
Có.

Ràng buộc bắt buộc triển khai P14 chắc chắn làm giảm không gian lựa chọn tối ưu,
vì mô hình buộc phải dành một phần ngân sách cho dự án an ninh mạng
ngay cả khi một số dự án khác có NPV kinh tế cao hơn.

Tuy nhiên,đây là một ràng buộc cực kỳ hợp lý dưới góc nhìn quản trị quốc gia.

Trong nền kinh tế số,an ninh mạng là điều kiện nền tảng bắt buộc,
không thể đánh giá đơn thuần bằng lợi ích tài chính trực tiếp.

Nếu thiếu đầu tư an ninh mạng,
toàn bộ:
- Dữ liệu quốc gia
- Hạ tầng số
- Hệ thống AI
- Chính phủ điện tử
đều có thể đối mặt với rủi ro tấn công mạng quy mô lớn.

Do đó,
P14 mang bản chất của một dự án phòng thủ chiến lược quốc gia,
tương tự như chi tiêu quốc phòng hoặc an ninh năng lượng.

Nói cách khác:
một phần chi tiêu công nghệ số phải được xem là “chi phí bảo hiểm quốc gia”,
không thể đánh giá chỉ bằng NPV kinh tế ngắn hạn.
""")

# --------------------------------------------------

st.subheader("""c) Mô hình giả định các dự án độc lập về lợi ích,nhưng trên thực tế P8 (AI quốc gia) và P13 (bán dẫn)
có lợi ích cộng hưởng. Làm thế nào để mô hình hóa hiệu ứng cộng hưởng này?""")

st.markdown("""
Trong thực tế,
P8 và P13 không hoàn toàn độc lập.

Nếu Việt Nam vừa phát triển trung tâm AI quốc gia,
vừa xây dựng công nghiệp bán dẫn,
hai dự án này sẽ tạo ra hiệu ứng cộng hưởng rất mạnh:
- AI làm tăng nhu cầu chip
- Bán dẫn giúp giảm phụ thuộc công nghệ nước ngoài
- Đồng thời thúc đẩy hệ sinh thái đổi mới sáng tạo nội địa.

Mô hình tuyến tính hiện tại chưa phản ánh được mối quan hệ này.

Để mô hình hóa hiệu ứng cộng hưởng,có thể bổ sung một biến tương tác:

z = y8 × y13

và cộng thêm một khoản lợi ích cộng hưởng:

Bonus × z

vào hàm mục tiêu.

Khi đó:
- Nếu chỉ có một dự án được chọn → không có bonus
- Nếu cả hai cùng được chọn → mô hình nhận thêm lợi ích chiến lược.

Đây là cách tiếp cận phổ biến trong:
- Quy hoạch công nghiệp
- Tối ưu danh mục đầu tư
- Và mô hình hóa externalities trong kinh tế học.

Việc đưa hiệu ứng cộng hưởng vào mô hình sẽ giúp kết quả phản ánh thực tế chính sách tốt hơn,
thay vì xem các dự án công nghệ chiến lược là hoàn toàn tách biệt.
""")
st.success("Hoàn thành toàn bộ bài 5.")