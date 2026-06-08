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
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
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
# CONFIG GIAO DIỆN STREAMLIT
# ==================================================


st.title("📊 Bài 4 — Quy hoạch tuyến tính phân bổ ngân sách số theo ngành - vùng")
# ==================================================
# SIDEBAR ĐIỀU CHỈNH CHÍNH SÁCH
# ==================================================

st.sidebar.header("⚙️ Điều chỉnh mô hình")

total_budget = st.sidebar.slider(
    "Ngân sách tổng (tỷ VND)",
    min_value=30000,
    max_value=100000,
    value=50000,
    step=1000
)

min_region_budget = st.sidebar.slider(
    "Sàn mỗi vùng",
    min_value=1000,
    max_value=10000,
    value=5000,
    step=500
)

max_region_budget = st.sidebar.slider(
    "Trần mỗi vùng",
    min_value=6000,
    max_value=20000,
    value=12000,
    step=500
)

min_human_budget = st.sidebar.slider(
    "Sàn nhân lực số",
    min_value=5000,
    max_value=30000,
    value=12000,
    step=500
)

gamma = st.sidebar.slider(
    "Gamma công bằng",
    min_value=0.000,
    max_value=0.010,
    value=0.002,
    step=0.001,
    format="%.3f"
)

lam = st.sidebar.slider(
    "Lambda công bằng",
    min_value=0.50,
    max_value=1.00,
    value=0.70,
    step=0.05
)

use_fairness = st.sidebar.checkbox(
    "Bật ràng buộc công bằng C5",
    value=True
)
st.sidebar.markdown("---")

st.sidebar.metric(
    "Ngân sách",
    f"{total_budget:,}"
)

st.sidebar.metric(
    "Sàn vùng",
    f"{min_region_budget:,}"
)

st.sidebar.metric(
    "Trần vùng",
    f"{max_region_budget:,}"
)

st.sidebar.metric(
    "Lambda",
    f"{lam:.2f}"
)
# ==================================================
# 4.1 & 4.2 — BỐI CẢNH VÀ MÔ HÌNH TOÁN HỌC
# ==================================================


# ==================================================
# 4.1 & 4.2 — BỐI CẢNH VÀ MÔ HÌNH TOÁN HỌC
# ==================================================


st.header("4.1. Bối cảnh Việt Nam")


st.markdown("""
Theo Quyết định số 411/QĐ-TTg ngày 31/3/2022 về phê duyệt Chiến lược quốc gia
phát triển kinh tế số và xã hội số, các vùng kinh tế - xã hội của Việt Nam
hiện có mức độ sẵn sàng số rất khác nhau.


Trong bối cảnh đó, bài toán chính sách được đặt ra là:


Phân bổ **50.000 tỷ VND** ngân sách kinh tế số quốc gia cho 6 vùng kinh tế
và 4 hạng mục đầu tư chiến lược gồm:


- **I** — Hạ tầng số
- **D** — Chuyển đổi số doanh nghiệp
- **AI** — Năng lực trí tuệ nhân tạo
- **H** — Nhân lực số và an sinh số


Mục tiêu của Chính phủ là vừa tối đa hóa mức tăng trưởng GDP kỳ vọng,
vừa bảo đảm tính công bằng vùng miền trong tiến trình chuyển đổi số quốc gia.
""")


# ==================================================
# 4.2 MÔ HÌNH TOÁN HỌC
# ==================================================


st.header("4.2. Mô hình toán học đầy đủ")


# ==================================================
# BIẾN QUYẾT ĐỊNH
# ==================================================


st.subheader("Biến quyết định")


st.latex(r"x_{j,r} \in \mathbb{R}^{+}")


st.markdown("""
Trong đó:


- \(j \in \{I, D, AI, H\}\) là các hạng mục đầu tư số
- \(r \in \{1,2,3,4,5,6\}\) là các vùng kinh tế - xã hội


Tổng cộng mô hình có **24 biến quyết định chính**.


Trong phiên bản mở rộng, có thể chia tiếp theo ngành kinh tế
để mở rộng quy mô mô hình lên hàng trăm biến.
""")


# ==================================================
# HÀM MỤC TIÊU
# ==================================================


st.subheader("Hàm mục tiêu tối ưu hóa GDP gain")


st.latex(
    r"""
    \max Z = \sum_{r}\sum_{j} \beta_{j,r} \cdot x_{j,r}
    """
)


st.markdown("""
Trong đó:


- \(x_{j,r}\): lượng vốn đầu tư cho hạng mục \(j\) tại vùng \(r\)
- \(\beta_{j,r}\): hệ số tác động biên của đầu tư tới GDP


Hệ số \(\beta_{j,r}\) phản ánh mức độ hiệu quả của từng loại đầu tư
tại từng vùng.


Ví dụ:


- Các vùng phát triển mạnh như Đông Nam Bộ thường có hệ số AI cao
- Các vùng khó khăn lại có hệ số đầu tư nhân lực và an sinh cao hơn
do hiệu ứng học tập lớn
""")


# ==================================================
# CÁC RÀNG BUỘC
# ==================================================


st.subheader("Các nhóm ràng buộc của mô hình")


# --------------------------------------------------
# C1
# --------------------------------------------------


st.markdown("### (C1) Ràng buộc ngân sách tổng")


st.latex(
    r"""
    \sum_{r}\sum_{j} x_{j,r} \leq 50{,}000
    """
)


st.markdown("""
Tổng ngân sách đầu tư số quốc gia không vượt quá 50.000 tỷ VND.
""")


# --------------------------------------------------
# C2
# --------------------------------------------------


st.markdown("### (C2) Sàn ngân sách tối thiểu cho mỗi vùng")


st.latex(
    r"""
    \sum_{j} x_{j,r} \geq 5{,}000 \quad \forall r
    """
)


st.markdown("""
Ràng buộc này nhằm tránh tình trạng dòng vốn chỉ tập trung vào các vùng giàu,
đồng thời bảo đảm mọi địa phương đều có nguồn lực tối thiểu
để tham gia chuyển đổi số.
""")


# --------------------------------------------------
# C3
# --------------------------------------------------


st.markdown("### (C3) Trần ngân sách cho mỗi vùng")


st.latex(
    r"""
    \sum_{j} x_{j,r} \leq 12{,}000 \quad \forall r
    """
)


st.markdown("""
Chính sách này giúp hạn chế hiện tượng các cực tăng trưởng như
Đồng bằng sông Hồng hoặc Đông Nam Bộ hấp thụ quá mức nguồn lực quốc gia.
""")


# --------------------------------------------------
# C4
# --------------------------------------------------


st.markdown("### (C4) Sàn đầu tư cho nhân lực số")


st.latex(
    r"""
    \sum_{r} x_{H,r} \geq 12{,}000
    """
)


st.markdown("""
Ít nhất 24% tổng ngân sách phải được phân bổ cho phát triển nhân lực số,
giáo dục số và an sinh số nhằm bảo đảm tính bền vững
của chuyển đổi số quốc gia.
""")


# --------------------------------------------------
# C5
# --------------------------------------------------


st.markdown("### (C5) Ràng buộc công bằng vùng miền")


st.latex(
    r"""
    D_r + \gamma x_{D,r}
    \geq
    \lambda \cdot
    \max_r(D_r + \gamma x_{D,r})
    """
)


st.markdown("""
Trong đó:


- \(D_r\): chỉ số Digital Index ban đầu của vùng
- \(\gamma = 0.002\): hệ số tác động của đầu tư nhân lực số
- \(\lambda = 0.7\): ngưỡng công bằng tối thiểu


Ràng buộc này bảo đảm rằng không vùng nào bị tụt quá xa
so với vùng dẫn đầu về năng lực số sau đầu tư.
""")


# --------------------------------------------------
# C6
# --------------------------------------------------


st.markdown("### (C6) Điều kiện không âm")


st.latex(
    r"""
    x_{j,r} \geq 0
    """
)


st.markdown("""
Mọi khoản đầu tư đều phải không âm.


Mô hình không cho phép phân bổ ngân sách âm
hoặc rút vốn khỏi một vùng kinh tế.
""")


# ==================================================
# 4.4.1 — TỐI ƯU HÓA BẰNG PULP
# ==================================================
# ==================================================
# DỮ LIỆU MÔ HÌNH
# ==================================================


regions = [
    "NMM",
    "RRD",
    "NCC",
    "CH",
    "SE",
    "MD"
]


regions_full = {
    "NMM": "Trung du miền núi phía Bắc",
    "RRD": "Đồng bằng sông Hồng",
    "NCC": "Bắc Trung Bộ và Duyên hải miền Trung",
    "CH": "Tây Nguyên",
    "SE": "Đông Nam Bộ",
    "MD": "Đồng bằng sông Cửu Long"
}


items = [
    "I",
    "D",
    "AI",
    "H"
]


items_full = {
    "I": "Hạ tầng số",
    "D": "Chuyển đổi số doanh nghiệp",
    "AI": "Năng lực AI",
    "H": "Nhân lực số"
}


# Hệ số beta
beta = {


    ("NMM","I"):1.15,
    ("NMM","D"):0.85,
    ("NMM","AI"):0.55,
    ("NMM","H"):1.30,


    ("RRD","I"):0.95,
    ("RRD","D"):1.25,
    ("RRD","AI"):1.40,
    ("RRD","H"):1.05,


    ("NCC","I"):1.05,
    ("NCC","D"):0.95,
    ("NCC","AI"):0.85,
    ("NCC","H"):1.15,


    ("CH","I"):1.20,
    ("CH","D"):0.75,
    ("CH","AI"):0.45,
    ("CH","H"):1.35,


    ("SE","I"):0.90,
    ("SE","D"):1.30,
    ("SE","AI"):1.55,
    ("SE","H"):1.00,


    ("MD","I"):1.10,
    ("MD","D"):0.85,
    ("MD","AI"):0.65,
    ("MD","H"):1.25
}


# Digital Index ban đầu


D0 = {
    "NMM": 38,
    "RRD": 78,
    "NCC": 55,
    "CH": 32,
    "SE": 82,
    "MD": 48
}






st.subheader("4.4.1 Kết quả tối ưu hóa bằng PuLP")


# Khởi tạo mô hình
prob_pulp = pulp.LpProblem(
    "VN_Digital_Budget",
    pulp.LpMaximize
)


# Biến quyết định
x_pulp = pulp.LpVariable.dicts(
    "x",
    (regions, items),
    lowBound=0
)


# Biến max Digital Index
M_pulp = pulp.LpVariable("D_max")


# --------------------------------------------------
# Hàm mục tiêu
# --------------------------------------------------


prob_pulp += pulp.lpSum(
    beta[(r, j)] * x_pulp[r][j]
    for r in regions
    for j in items
)


# --------------------------------------------------
# Ràng buộc C1
# --------------------------------------------------


prob_pulp += pulp.lpSum(
    x_pulp[r][j]
    for r in regions
    for j in items
) <= total_budget


# --------------------------------------------------
# Ràng buộc C2 và C3
# --------------------------------------------------


for r in regions:


    prob_pulp += pulp.lpSum(
        x_pulp[r][j]
        for j in items
    ) >= min_region_budget


    prob_pulp += pulp.lpSum(
        x_pulp[r][j]
        for j in items
    ) <= max_region_budget


# --------------------------------------------------
# Ràng buộc C4
# --------------------------------------------------


prob_pulp += pulp.lpSum(
    x_pulp[r]["H"]
    for r in regions
) >= min_human_budget


# --------------------------------------------------
# Ràng buộc C5
# --------------------------------------------------

if use_fairness:

    for r in regions:
        prob_pulp += (
            D0[r] + gamma * x_pulp[r]["D"]
            <= M_pulp
        )

    for r in regions:
        prob_pulp += (
            D0[r] + gamma * x_pulp[r]["D"]
            >= lam * M_pulp
        )



# --------------------------------------------------
# Giải mô hình
# --------------------------------------------------


prob_pulp.solve(
    pulp.PULP_CBC_CMD(msg=False)
)


Z_pulp = pulp.value(prob_pulp.objective)


# --------------------------------------------------
# Tạo bảng kết quả
# --------------------------------------------------


matrix_pulp = np.zeros((6, 4))


for i, r in enumerate(regions):
    for j, item in enumerate(items):


        value = x_pulp[r][item].varValue


        if value is None:
            value = 0


        matrix_pulp[i, j] = value


df_pulp = pd.DataFrame(
    matrix_pulp,
    index=[regions_full[r] for r in regions],
    columns=[items_full[j] for j in items]
)


# --------------------------------------------------
# Hiển thị kết quả
# --------------------------------------------------


st.success(
    f"Giá trị GDP gain tối ưu: {Z_pulp:,.2f} tỷ VND"
)


st.markdown(
    "### Ma trận phân bổ ngân sách tối ưu"
)


st.dataframe(
    df_pulp.style.format("{:,.2f}"),
    use_container_width=True
)


# ==================================================
# 4.4.2 — CVXPY VÀ SO SÁNH
# ==================================================


st.subheader("4.4.2 So sánh PuLP và CVXPY")


# --------------------------------------------------
# Khởi tạo CVXPY
# --------------------------------------------------


x_cvx = cp.Variable((6, 4), nonneg=True)


M_cvx = cp.Variable()


# --------------------------------------------------
# Ma trận beta
# --------------------------------------------------


beta_matrix = np.zeros((6, 4))


for i, r in enumerate(regions):
    for j, item in enumerate(items):


        beta_matrix[i, j] = beta[(r, item)]


D0_arr = np.array([D0[r] for r in regions])


# --------------------------------------------------
# Hàm mục tiêu
# --------------------------------------------------


objective = cp.Maximize(
    cp.sum(
        cp.multiply(beta_matrix, x_cvx)
    )
)


# --------------------------------------------------
# Ràng buộc
# --------------------------------------------------


constraints = [


    cp.sum(x_cvx) <= total_budget,


    cp.sum(x_cvx, axis=1) >= min_region_budget,


    cp.sum(x_cvx, axis=1) <= max_region_budget,


    cp.sum(x_cvx[:, 3])  >= min_human_budget
]


# C5
if use_fairness:

    for i in range(6):


     constraints.append(
        D0_arr[i] + gamma * x_cvx[i, 1]
        <= M_cvx
    )


    constraints.append(
        D0_arr[i] + gamma * x_cvx[i, 1]
        >= lam * M_cvx
    )


# --------------------------------------------------
# Giải bài toán
# --------------------------------------------------


prob_cvx = cp.Problem(
    objective,
    constraints
)


prob_cvx.solve()


Z_cvx = prob_cvx.value


diff_Z = abs(Z_pulp - Z_cvx)


# --------------------------------------------------
# Hiển thị kết quả
# --------------------------------------------------


col1, col2 = st.columns(2)


with col1:


    st.metric(
        "PuLP",
        f"{Z_pulp:,.2f}"
    )


    st.metric(
        "CVXPY",
        f"{Z_cvx:,.2f}"
    )


with col2:


    st.metric(
        "Sai số tuyệt đối",
        f"{diff_Z:,.6f}"
    )


    if diff_Z < 0.01:


        st.success(
            "Hai phương pháp cho kết quả gần như giống nhau."
        )


    else:


        st.warning(
            "Có sai lệch nhỏ giữa hai bộ giải."
        )


# ==================================================
# 4.4.3 — HEATMAP
# ==================================================


st.subheader("4.4.3 Heatmap phân bổ tối ưu")


fig, ax = plt.subplots(
    figsize=(12, 6)
)


sns.heatmap(
    df_pulp,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    linewidths=0.5,
    ax=ax
)


ax.set_title(
    "Heatmap phân bổ ngân sách số"
)


ax.set_xlabel(
    "Hạng mục đầu tư"
)


ax.set_ylabel(
    "Vùng kinh tế"
)


plt.xticks(rotation=15)


st.pyplot(fig)


st.markdown("""
### Nhận xét


- Đông Nam Bộ và Đồng bằng sông Hồng nhận lượng vốn lớn nhất.
- Các vùng phát triển mạnh ưu tiên AI và nhân lực số.
- Các vùng khó khăn ưu tiên hạ tầng và an sinh số.
""")


# ==================================================
# 4.4.4 — BỎ RÀNG BUỘC CÔNG BẰNG
# ==================================================


st.subheader("4.4.4 So sánh với mô hình bỏ C5")


# --------------------------------------------------
# Mô hình không có C5
# --------------------------------------------------


prob_no_c5 = pulp.LpProblem(
    "No_C5",
    pulp.LpMaximize
)


x_no_c5 = pulp.LpVariable.dicts(
    "x_no_c5",
    (regions, items),
    lowBound=0
)


# Hàm mục tiêu
prob_no_c5 += pulp.lpSum(
    beta[(r, j)] * x_no_c5[r][j]
    for r in regions
    for j in items
)


# C1
prob_no_c5 += pulp.lpSum(
    x_no_c5[r][j]
    for r in regions
    for j in items
) <= 50000


# C2, C3
for r in regions:


    prob_no_c5 += pulp.lpSum(
        x_no_c5[r][j]
        for j in items
    ) >= 5000


    prob_no_c5 += pulp.lpSum(
        x_no_c5[r][j]
        for j in items
    ) <= 12000


# C4
prob_no_c5 += pulp.lpSum(
    x_no_c5[r]["H"]
    for r in regions
) >= 12000


# --------------------------------------------------
# Giải mô hình
# --------------------------------------------------


prob_no_c5.solve(
    pulp.PULP_CBC_CMD(msg=False)
)


Z_no_c5 = pulp.value(
    prob_no_c5.objective
)


gdp_loss = Z_no_c5 - Z_pulp


# --------------------------------------------------
# Hiển thị
# --------------------------------------------------


col3, col4 = st.columns(2)


with col3:


    st.metric(
        "Có C5",
        f"{Z_pulp:,.2f}"
    )


    st.metric(
        "Không có C5",
        f"{Z_no_c5:,.2f}"
    )


with col4:


    st.metric(
        "Chi phí công bằng",
        f"{gdp_loss:,.2f} tỷ VND"
    )


st.info("""
Ràng buộc công bằng giúp hạn chế chênh lệch số hóa giữa các vùng,
nhưng đồng thời làm giảm một phần GDP gain tối đa.
""")


# ==================================================
# 4.5. CÂU HỎI THẢO LUẬN CHÍNH SÁCH
# ==================================================


st.header("4.5. Câu hỏi thảo luận chính sách")


# ==================================================
# CÂU A
# ==================================================


st.subheader(
    "a) Nếu bỏ ràng buộc công bằng, vốn sẽ chảy về vùng nào? "
    "Tại sao? Hậu quả xã hội dài hạn ra sao?"
)


st.markdown("""


Khi loại bỏ hoàn toàn ràng buộc công bằng vùng miền (C5),
mô hình tối ưu sẽ có xu hướng dồn mạnh ngân sách về:


- Đông Nam Bộ (SE)
- Đồng bằng sông Hồng (RRD)


Đây là hai vùng có hiệu suất sinh lời kinh tế cao nhất,
đặc biệt trong các lĩnh vực:


- AI Applications
- Chuyển đổi số doanh nghiệp
- Nhân lực số


Ví dụ:


- \\(\\beta_{SE,AI} = 1.55\\)
- \\(\\beta_{RRD,AI} = 1.40\\)


Điều này có nghĩa rằng:
mỗi 1 tỷ VND đầu tư vào AI tại các vùng này
tạo ra mức tăng GDP cao hơn đáng kể so với các vùng còn lại.


Do mục tiêu của mô hình là tối đa hóa GDP gain,
nguồn vốn sẽ tự động chảy về nơi có hiệu quả kinh tế cao nhất.


---
### Hậu quả xã hội dài hạn


Nếu không tồn tại cơ chế công bằng vùng miền,
quá trình này có thể dẫn đến:


- Khoảng cách số hóa giữa các vùng ngày càng lớn
- Các vùng khó khăn bị tụt hậu công nghệ
- Gia tăng chênh lệch giàu nghèo
- Dịch chuyển lao động chất lượng cao về các đô thị lớn
- Áp lực dân số và hạ tầng tại các cực tăng trưởng


Về dài hạn,
nền kinh tế có thể đạt GDP cao hơn,
nhưng đánh đổi bằng sự mất cân bằng phát triển vùng miền.
""")


# ==================================================
# CÂU B
# ==================================================


st.subheader(
    "b) Ràng buộc trần ngân sách mỗi vùng (C3) có thể coi như "
    "một “chính sách phân quyền”. Nó làm giảm Z* bao nhiêu "
    "phần trăm? Mức giảm này có chấp nhận được không?"
)


# --------------------------------------------------
# GIẢI MÔ HÌNH KHÔNG CÓ C3
# --------------------------------------------------


prob_no_c3 = pulp.LpProblem(
    "VN_Digital_Budget_No_C3",
    pulp.LpMaximize
)


x_no_c3 = pulp.LpVariable.dicts(
    "x_no_c3",
    (regions, items),
    lowBound=0
)


M_no_c3 = pulp.LpVariable("Dmax_no_c3")


# Hàm mục tiêu
prob_no_c3 += pulp.lpSum(
    beta[(r, j)] * x_no_c3[r][j]
    for r in regions
    for j in items
)


# C1
prob_no_c3 += pulp.lpSum(
    x_no_c3[r][j]
    for r in regions
    for j in items
) <= 50000


# C2
for r in regions:
    prob_no_c3 += pulp.lpSum(
        x_no_c3[r][j]
        for j in items
    ) >= 5000


# C4
prob_no_c3 += pulp.lpSum(
    x_no_c3[r]["H"]
    for r in regions
) >= 12000


# C5
for r in regions:
    prob_no_c3 += (
        D0[r] + gamma * x_no_c3[r]["D"]
        <= M_no_c3
    )


for r in regions:
    prob_no_c3 += (
        D0[r] + gamma * x_no_c3[r]["D"]
        >= lam * M_no_c3
    )


# Giải mô hình
prob_no_c3.solve(pulp.PULP_CBC_CMD(msg=False))


Z_no_c3 = pulp.value(prob_no_c3.objective)


# Tính phần trăm giảm GDP
pct_loss_c3 = (
    (Z_no_c3 - Z_pulp) / Z_no_c3
) * 100


st.markdown(f"""


Ràng buộc C3 giới hạn tổng ngân sách tối đa cho mỗi vùng ở mức:


- 12.000 tỷ VND


Chính sách này giúp tránh tình trạng
một vài vùng phát triển mạnh hấp thụ gần như toàn bộ nguồn lực quốc gia.


---
### Kết quả mô phỏng


- GDP tối ưu khi có C3:
  **{Z_pulp:.2f} tỷ VND**


- GDP tối ưu khi bỏ C3:
  **{Z_no_c3:.2f} tỷ VND**


- Mức suy giảm của \\(Z^*\\):
  **{pct_loss_c3:.2f}%**


---
### Đánh giá chính sách


Mức giảm hiệu quả kinh tế này là tương đối nhỏ
so với lợi ích xã hội mà chính sách mang lại.


Ràng buộc C3 giúp:


- Hạn chế tập trung vốn quá mức
- Bảo đảm mọi vùng đều có cơ hội chuyển đổi số
- Giảm nguy cơ phân hóa vùng miền
- Tăng tính bền vững cho chiến lược phát triển quốc gia


Vì vậy,
mức đánh đổi GDP để đổi lấy công bằng vùng miền
được xem là hợp lý và chấp nhận được.
""")


# ==================================================
# CÂU C
# ==================================================


st.subheader(
    "c) Vùng Tây Nguyên có sàn 5.000 tỷ nhưng hệ số AI rất thấp (0,45). "
    "Nên đầu tư vào AI tại Tây Nguyên hay tập trung H và I trước? "
    "Mô hình trả lời như thế nào?"
)


st.markdown("""


Kết quả từ mô hình tối ưu cho thấy:


Tây Nguyên chưa phải là khu vực phù hợp
để ưu tiên đầu tư mạnh vào AI trong giai đoạn hiện tại.


Nguyên nhân là vì:


- Hệ số hiệu quả đầu tư AI của vùng khá thấp:
  \\(\\beta_{CH,AI} = 0.45\\)


- Nền tảng hạ tầng số còn hạn chế


- Nguồn nhân lực số chưa đủ mạnh


- Khả năng hấp thụ công nghệ AI còn thấp


Trong khi đó:


- Hạ tầng số (I) có hệ số sinh lời cao:
  \\(\\beta_{CH,I} = 1.20\\)


- Nhân lực số và an sinh số (H) có hiệu quả rất tốt:
  \\(\\beta_{CH,H} = 1.35\\)


Do đó,
mô hình lựa chọn ưu tiên phân bổ ngân sách cho:


- Hạ tầng số
- Giáo dục số
- Chính phủ số
- Nhân lực số


trước khi mở rộng đầu tư AI quy mô lớn.


---
### Kết luận chính sách


Mô hình cho thấy chiến lược phù hợp với Tây Nguyên là:


1. Phát triển hạ tầng số trước
2. Nâng cao chất lượng nhân lực số
3. Tăng mức độ sẵn sàng số của vùng
4. Sau đó mới mở rộng đầu tư AI


Điều này phản ánh đúng logic phát triển kinh tế số:


AI chỉ phát huy hiệu quả mạnh
khi địa phương đã có nền tảng hạ tầng và nhân lực đủ tốt
để hấp thụ công nghệ.
""")


st.success("Hoàn thành toàn bộ bài 4.")

# =========================================================
# OUTPUT CHO BÀI 12 - AIDEOM VN
# =========================================================

st.markdown("---")
st.header("📤 Output cho Bài 12 — AIDEOM-VN Dashboard")

# =========================================================
# OUTPUT MODULE BÀI 4
# =========================================================

M3_OUTPUT_B4 = {

    # =====================================
    # THÔNG TIN MÔ HÌNH
    # =====================================
    "model_info": {

        "model_name":
        "Digital Budget Allocation LP",

        "objective":
        "Maximize GDP Gain",

        "total_budget":
        50000,

        "regions":
        regions,

        "investment_categories":
        items
    },

    # =====================================
    # KẾT QUẢ PULP
    # =====================================
    "pulp_solution": {

        "GDP_gain_optimal":
        round(Z_pulp, 2),

        "budget_allocation": {

            r: {

                j: round(
                    x_pulp[r][j].varValue, 2
                )

                for j in items
            }

            for r in regions
        }
    },

    # =====================================
    # KẾT QUẢ CVXPY
    # =====================================
    "cvxpy_solution": {

        "GDP_gain_cvx":
        round(Z_cvx, 2),

        "absolute_error":
        round(diff_Z, 6)
    },

    # =====================================
    # FAIRNESS ANALYSIS
    # =====================================
    "fairness_analysis": {

        "GDP_without_C5":
        round(Z_no_c5, 2),

        "fairness_cost":
        round(gdp_loss, 2)
    },

    # =====================================
    # DECENTRALIZATION ANALYSIS
    # =====================================
    "decentralization_analysis": {

        "GDP_without_C3":
        round(Z_no_c3, 2),

        "GDP_loss_percent":
        round(pct_loss_c3, 2)
    },

    # =====================================
    # DIGITAL INDEX
    # =====================================
    "digital_index": {

        r: D0[r]
        for r in regions
    },

    # =====================================
    # BETA COEFFICIENTS
    # =====================================
    "beta_coefficients": {

        f"{r}_{j}":
        beta[(r, j)]

        for r in regions
        for j in items
    }
}

# =========================================================
# LƯU SESSION_STATE CHO BÀI 12
# =========================================================

st.session_state["M3_OUTPUT_B4"] = M3_OUTPUT_B4

# =========================================================
# HIỂN THỊ JSON
# =========================================================

def get_b4_output():
    return M3_OUTPUT_B4


if __name__ == "__main__":
    st.json(get_b4_output())

st.success("""
✅ Bài 4 Output đã sẵn sàng để tích hợp vào:
Bài 12 — AIDEOM-VN Dashboard
""")