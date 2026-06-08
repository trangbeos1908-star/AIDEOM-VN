# =========================================================
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

# =========================================================
# TITLE
# =========================================================
st.title("📊 Bài 7—Tối ưu đa mục tiêu Pareto với NSGA-II ")
st.sidebar.header("⚙️ Điều chỉnh NSGA-II")
risk_factor = st.sidebar.slider(
    "Mức rủi ro AI",
    0.5,
    2.0,
    1.0,
    0.1
)

security_factor = st.sidebar.slider(
    "Hiệu quả đào tạo",
    0.5,
    2.0,
    1.0,
    0.1
)

pop_size = st.sidebar.slider(
    "Population Size",
    20, 300, 100, 10
)

n_gen = st.sidebar.slider(
    "Number of Generations",
    50, 1000, 200, 50
)

total_budget = st.sidebar.slider(
    "Ngân sách tổng",
    30000, 100000, 50000, 1000
)

min_region_budget = st.sidebar.slider(
    "Sàn mỗi vùng",
    1000, 10000, 5000, 500
)

max_region_budget = st.sidebar.slider(
    "Trần mỗi vùng",
    6000, 20000, 12000, 500
)

min_h_budget = st.sidebar.slider(
    "Ngân sách nhân lực tối thiểu",
    5000, 30000, 12000, 500
)
st.sidebar.header("🎯 Trọng số TOPSIS")

w_growth = st.sidebar.slider(
    "Tăng trưởng",
    0.0, 1.0, 0.40, 0.05
)

w_inclusion = st.sidebar.slider(
    "Bao trùm",
    0.0, 1.0, 0.25, 0.05
)

w_environment = st.sidebar.slider(
    "Môi trường",
    0.0, 1.0, 0.20, 0.05
)

w_security = st.sidebar.slider(
    "An ninh dữ liệu",
    0.0, 1.0, 0.15, 0.05
)
if min_region_budget * 6 > total_budget:
    st.error("Tổng sàn ngân sách vùng vượt ngân sách tổng.")
    st.stop()

if max_region_budget * 6 < total_budget:
    st.warning("Tổng trần vùng nhỏ hơn ngân sách tổng, mô hình có thể không dùng hết ngân sách.")
st.subheader("7.1. Bối cảnh Việt Nam")

st.markdown("""
Mục 8.2 của bài báo nguồn nhấn mạnh rằng kết quả của bài toán tối ưu hóa phát triển kinh tế trong kỷ nguyên trí tuệ
nhân tạo không nhất thiết phải là một nghiệm tối ưu duy nhất, mà có thể là một tập nghiệm Pareto.

Đây là một luận điểm có ý nghĩa quan trọng đối với hoạch định chính sách công, bởi nó thừa nhận rằng các mục tiêu
phát triển thường tồn tại quan hệ đánh đổi lẫn nhau. Do đó, lựa chọn chính sách không thể chỉ dựa trên tính toán kỹ
thuật thuần túy, mà còn cần được đặt trong quá trình thảo luận chính trị, xã hội và thể chế.

Trong bối cảnh Việt Nam, quá trình phát triển kinh tế số và ứng dụng AI đang đồng thời hướng tới bốn mục tiêu
chiến lược:

(i) Thúc đẩy tăng trưởng GDP nhanh

(ii) Bảo đảm tính bao trùm xã hội, đặc biệt thông qua giảm bất bình đẳng vùng và bất bình đẳng số

(iii) Thực hiện mục tiêu phát thải ròng bằng 0 vào năm 2050 theo cam kết tại COP26

(iv) Tăng cường an ninh dữ liệu và bảo vệ chủ quyền số quốc gia.
""")
st.subheader("7.2. Mô hình toán học đa mục tiêu")

st.markdown(r"""
Trên cơ sở dữ liệu của Bài 4, bài tập này yêu cầu sinh viên xây dựng và giải một mô hình tối ưu hóa đa mục tiêu gồm
bốn hàm mục tiêu tương ứng với các định hướng chính sách phát triển của Việt Nam.

Gọi:

- \( x_{j,r} \geq 0 \) là mức ngân sách phân bổ cho hạng mục đầu tư \( j \) tại vùng \( r \)

Trong đó:

- \( j \in \{I, D, AI, H\} \)
- \( r \in \{1,2,\dots,6\} \)

Như vậy, mô hình có tổng cộng:

\[
4 \times 6 = 24
\]

biến quyết định.

Các hạng mục đầu tư bao gồm:

- Hạ tầng số (I)
- Dữ liệu (D)
- Trí tuệ nhân tạo (AI)
- Phát triển nguồn nhân lực (H)
""")

st.markdown("### Mục tiêu 1: Tối đa hóa tăng trưởng GDP kỳ vọng")

st.latex(r"""
\max f_1(x)=\sum_r \sum_j \beta_{j,r} x_{j,r}
""")

st.markdown(r"""
Trong đó:

- \( \beta_{j,r} \) là hệ số tác động cận biên của đầu tư vào hạng mục \( j \) tại vùng \( r \)
- Hàm mục tiêu phản ánh đóng góp kỳ vọng vào tăng trưởng GDP
""")

st.markdown("### Mục tiêu 2: Giảm bất bình đẳng phân bổ ngân sách")

st.latex(r"""
\min f_2(x)=G(x)
""")

st.markdown(r"""
Trong đó:

- \( G(x) \) là chỉ số Gini hoặc đại lượng xấp xỉ Gini
- Hàm mục tiêu phản ánh mức độ công bằng trong phân bổ ngân sách giữa các vùng
""")

st.markdown("### Mục tiêu 3: Giảm phát thải gián tiếp")

st.latex(r"""
\min f_3(x)=\sum_r e_r (x_{I,r}+x_{AI,r})
""")

st.markdown(r"""
Trong đó:

- \( e_r \) là hệ số cường độ phát thải của vùng \( r \)
- Hàm mục tiêu phản ánh tác động môi trường của đầu tư hạ tầng số và AI
""")

st.markdown("### Mục tiêu 4: Giảm rủi ro an ninh dữ liệu")

st.latex(r"""
\min f_4(x)=\sum_r \rho_r x_{AI,r}-\sum_r \sigma_r x_{H,r}
""")

st.markdown(r"""
Trong đó:

- \( \rho_r \) là hệ số rủi ro dữ liệu do đầu tư AI
- \( \sigma_r \) là hệ số giảm thiểu rủi ro nhờ đầu tư nguồn nhân lực
""")

st.markdown("### Dạng tổng quát của mô hình")

st.latex(r"""
\max f_1(x), \quad
\min f_2(x), \quad
\min f_3(x), \quad
\min f_4(x)
""")

st.markdown(r"""
Mô hình chịu các ràng buộc:

- Ràng buộc ngân sách
- Năng lực triển khai
- Giới hạn vùng
- Bao trùm số
- Các điều kiện chính sách khác

Tương ứng với hệ ràng buộc:

\[
C_1 - C_6
\]

Do bốn mục tiêu có thể xung đột với nhau, kết quả của mô hình không phải là một nghiệm tối ưu duy nhất mà là
một tập nghiệm Pareto.

Tập nghiệm Pareto cho phép nhà hoạch định chính sách đánh giá các phương án phân bổ nguồn lực khác nhau nhằm
cân bằng giữa:

- Tăng trưởng kinh tế
- Công bằng xã hội
- Chuyển đổi xanh
- An ninh dữ liệu và chủ quyền số
""")
# =========================================================
# 7.4.1 BÀI TOÁN ĐA MỤC TIÊU
# =========================================================
st.subheader(
    "7.4.1. Cài đặt mô hình đa mục tiêu bằng pymoo và NSGA-II"
)

# =========================================================
# DANH SÁCH VÙNG
# =========================================================
regions = [
    "Trung du miền núi phía Bắc",
    "Đồng bằng sông Hồng",
    "Bắc Trung Bộ & DH miền Trung",
    "Tây Nguyên",
    "Đông Nam Bộ",
    "Đồng bằng sông Cửu Long"
]

# =========================================================
# DANH SÁCH HẠNG MỤC
# =========================================================
items = ["I", "D", "AI", "H"]

# =========================================================
# MA TRẬN β (GDP gain)
# Hàng = vùng
# Cột = I, D, AI, H
# =========================================================
beta = np.array([
    [0.85, 0.75, 0.45, 0.95],
    [1.35, 1.20, 1.55, 1.10],
    [1.00, 0.92, 0.80, 1.05],
    [0.72, 0.68, 0.45, 1.18],
    [1.48, 1.32, 1.70, 1.15],
    [0.90, 0.82, 0.62, 1.02]
])

# =========================================================
# THAM SỐ BỔ SUNG
# =========================================================
emission_factor = st.sidebar.slider(
    "Hệ số môi trường",
    0.5, 2.0, 1.0, 0.1
)
e = np.array([
    0.42,0.55,0.48,0.32,0.62,0.38
]) * emission_factor

rho = np.array([
    0.18,
    0.45,
    0.28,
    0.12,
    0.52,
    0.22
]) * risk_factor

sigma = np.array([
    0.32,
    0.28,
    0.30,
    0.35,
    0.25,
    0.30
]) * security_factor

# =========================================================
# CHỈ SỐ DIGITAL INDEX
# =========================================================
digital_index = np.array([
    38,
    78,
    55,
    32,
    82,
    48
])

# =========================================================
# CLASS PROBLEM
# =========================================================
class VietnamDigitalProblem(ElementwiseProblem):

    def __init__(self):

        super().__init__(
            n_var=24,
            n_obj=4,
            n_ieq_constr=14,
            xl=np.zeros(24),
            xu=np.ones(24) * 12000
        )

    def _evaluate(self, x, out, *args, **kwargs):

        X = x.reshape(6, 4)

        # =================================================
        # f1 = MAX GDP
        # pymoo luôn MIN
        # => đổi dấu âm
        # =================================================
        f1 = -(beta * X).sum()

        # =================================================
        # f2 = BAO TRÙM (MAD)
        # =================================================
        regional_sum = X.sum(axis=1)

        f2 = np.abs(
            regional_sum - regional_sum.mean()
        ).mean()

        # =================================================
        # f3 = PHÁT THẢI
        # chỉ tính I + AI
        # =================================================
        f3 = (
            e * (X[:, 0] + X[:, 2])
        ).sum()

        # =================================================
        # f4 = RỦI RO DỮ LIỆU
        # =================================================
        f4 = (
            rho * X[:, 2]
        ).sum() - (
            sigma * X[:, 3]
        ).sum()

        out["F"] = [f1, f2, f3, f4]

        # =================================================
        # RÀNG BUỘC
        # =================================================
        g = []

        # -------------------------------------------------
        # (C1) Tổng ngân sách <= 50.000
        # -------------------------------------------------
        g.append(X.sum() - total_budget)

        # -------------------------------------------------
        # (C2) Mỗi vùng >= 5.000
        # -------------------------------------------------
        for r in range(6):
            g.append(
    min_region_budget - X[r].sum()
)

        # -------------------------------------------------
        # (C3) Mỗi vùng <= 12.000
        # -------------------------------------------------
        for r in range(6):
            g.append(
    X[r].sum() - max_region_budget
)

        # -------------------------------------------------
        # (C4) H >= 12.000
        # -------------------------------------------------
        g.append(
    min_h_budget - X[:,3].sum()
)

        out["G"] = g

# =========================================================
# CHẠY NSGA-II
# =========================================================
problem = VietnamDigitalProblem()

algorithm = NSGA2(
    pop_size=pop_size
)

res = minimize(
    problem,
    algorithm,
    ("n_gen", n_gen),
    seed=42,
    verbose=False
)

# =========================================================
# KẾT QUẢ PARETO
# =========================================================
F = res.F

pareto_df = pd.DataFrame({
    "GDP_gain": -F[:, 0],
    "Bao_trum": F[:, 1],
    "Phat_thai": F[:, 2],
    "Rui_ro": F[:, 3]
})

st.success(
    f"Tìm được {len(pareto_df)} nghiệm Pareto tối ưu."
)

st.dataframe(
    pareto_df.head(20),
    use_container_width=True
)
# =========================================================
# PHÂN TÍCH KẾT QUẢ 7.4.1
# =========================================================

st.subheader("Kết quả chạy NSGA-II")

st.markdown("""
Sau khi chạy NSGA-II với:
- pop_size = 100
- n_gen = 200

mô hình thu được một tập nghiệm Pareto gồm nhiều phương án phân bổ ngân sách khác nhau.

### Nhận xét kết quả

Các nghiệm có:
- tăng trưởng GDP cao
thường đi kèm:
- phát thải CO₂ lớn hơn,
- và rủi ro dữ liệu cao hơn.

Ngược lại,
các nghiệm ưu tiên:
- công bằng vùng miền,
- và phát triển xanh
thường phải chấp nhận mức tăng trưởng thấp hơn.

### Ý nghĩa chính sách

Kết quả cho thấy tồn tại trade-off giữa:
- tăng trưởng kinh tế,
- công bằng xã hội,
- môi trường,
- và an ninh dữ liệu.

Do đó,
Chính phủ cần lựa chọn nghiệm phù hợp với ưu tiên chiến lược,
thay vì chỉ tối đa hóa GDP như mô hình tối ưu đơn mục tiêu truyền thống.
""")
# =========================================================
# 7.4.2 BIỂU ĐỒ 3D
# =========================================================
st.subheader(
    "7.4.2. Biểu đồ Pareto 3D và Parallel Coordinates"
)

# =========================================================
# SCATTER 3D
# =========================================================
fig = plt.figure(figsize=(10, 7))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    pareto_df["GDP_gain"],
    pareto_df["Bao_trum"],
    pareto_df["Phat_thai"]
)

ax.set_xlabel("GDP Gain")
ax.set_ylabel("Bao trùm")
ax.set_zlabel("Phát thải")

ax.set_title(
    "Pareto Front giữa tăng trưởng - bao trùm - môi trường"
)

st.pyplot(fig)

# =========================================================
# PARALLEL COORDINATES
# =========================================================
fig2, ax2 = plt.subplots(figsize=(12, 6))

for i in range(len(pareto_df)):
    ax2.plot(
        pareto_df.columns,
        pareto_df.iloc[i],
        alpha=0.35
    )

ax2.set_title(
    "Parallel Coordinates cho 4 mục tiêu"
)

st.pyplot(fig2)

# =========================================================
# 7.4.3 TOPSIS TRÊN PARETO
# =========================================================
st.subheader(
    "7.4.3. Chọn nghiệm thỏa hiệp bằng TOPSIS"
)

# =========================================================
# TRỌNG SỐ CHÍNH SÁCH
# =========================================================
weights = np.array([
    w_growth,
    w_inclusion,
    w_environment,
    w_security
])

if weights.sum() == 0:
    st.error("Tổng trọng số TOPSIS bằng 0. Hãy tăng ít nhất một trọng số.")
    st.stop()

weights = weights / weights.sum()

# =========================================================
# CHUẨN HÓA
# =========================================================
X_topsis = pareto_df.values

R = X_topsis / np.sqrt(
    (X_topsis ** 2).sum(axis=0)
)

# =========================================================
# TRỌNG SỐ
# =========================================================
V = R * weights

# =========================================================
# GDP là benefit
# các mục tiêu còn lại là cost
# =========================================================
benefit = [True, False, False, False]

A_star = np.zeros(4)
A_neg = np.zeros(4)

for j in range(4):

    if benefit[j]:
        A_star[j] = V[:, j].max()
        A_neg[j] = V[:, j].min()
    else:
        A_star[j] = V[:, j].min()
        A_neg[j] = V[:, j].max()

# =========================================================
# KHOẢNG CÁCH
# =========================================================
S_star = np.sqrt(
    ((V - A_star) ** 2).sum(axis=1)
)

S_neg = np.sqrt(
    ((V - A_neg) ** 2).sum(axis=1)
)

# =========================================================
# C_i*
# =========================================================
C_star = S_neg / (S_star + S_neg)

pareto_df["C_star"] = C_star

# =========================================================
# NGHIỆM THỎA HIỆP
# =========================================================
best_solution = pareto_df.sort_values(
    "C_star",
    ascending=False
).iloc[0]

st.write("### Nghiệm thỏa hiệp tốt nhất")

st.dataframe(
    pd.DataFrame(best_solution).T,
    use_container_width=True
)

# =========================================================
# PHÂN TÍCH
# =========================================================
st.markdown(f"""
Kết quả TOPSIS trên tập Pareto cho thấy nghiệm thỏa hiệp tốt nhất có:

- GDP Gain ≈ **{best_solution['GDP_gain']:.2f}**
- Bao trùm ≈ **{best_solution['Bao_trum']:.2f}**
- Phát thải ≈ **{best_solution['Phat_thai']:.2f}**
- Rủi ro ≈ **{best_solution['Rui_ro']:.2f}**
- Hệ số gần lý tưởng:
  **C* = {best_solution['C_star']:.4f}**

Nghiệm này đạt sự cân bằng tương đối tốt giữa:
- tăng trưởng kinh tế,
- công bằng vùng miền,
- giảm phát thải,
- và an ninh dữ liệu.

Điều này phản ánh đúng định hướng phát triển bền vững
trong chiến lược AI và kinh tế số quốc gia của Việt Nam.
""")

# =========================================================
# 7.4.4 CHI PHÍ CƠ HỘI
# =========================================================
st.subheader(
    "7.4.4. Phân tích chi phí cơ hội giữa các mục tiêu"
)

# =========================================================
# NGHIỆM GDP MAX
# =========================================================
growth_best = pareto_df.sort_values(
    "GDP_gain",
    ascending=False
).iloc[0]

# =========================================================
# TÍNH % HI SINH
# =========================================================
loss_baotrum = (
    (
        growth_best["Bao_trum"]
        - best_solution["Bao_trum"]
    )
    / best_solution["Bao_trum"]
) * 100

loss_environment = (
    (
        growth_best["Phat_thai"]
        - best_solution["Phat_thai"]
    )
    / best_solution["Phat_thai"]
) * 100

st.markdown(f"""
Nghiệm có tăng trưởng GDP cao nhất giúp tối đa hóa lợi ích kinh tế ngắn hạn,
nhưng phải đánh đổi đáng kể ở các mục tiêu khác.

### So với nghiệm thỏa hiệp:

- Mức bất bình đẳng phân bổ tăng khoảng:
  **{loss_baotrum:.2f}%**

- Mức phát thải tăng khoảng:
  **{loss_environment:.2f}%**

### Ý nghĩa chính sách

Điều này cho thấy:
nếu chỉ tập trung tối đa hóa GDP,
nguồn lực sẽ có xu hướng dồn vào các vùng có hiệu quả kinh tế cao như:
- Đông Nam Bộ,
- Đồng bằng sông Hồng.

Tuy nhiên:
- điều này có thể làm gia tăng chênh lệch vùng miền,
- tăng phát thải carbon,
- và làm suy giảm tính bền vững dài hạn.

Ngược lại,
nghiệm thỏa hiệp giúp cân bằng tốt hơn giữa:
- hiệu quả tăng trưởng,
- công bằng xã hội,
- và mục tiêu phát triển xanh.

Đây cũng là tinh thần cốt lõi của tối ưu hóa Pareto:
không tồn tại một nghiệm tốt nhất tuyệt đối,
mà luôn tồn tại các đánh đổi giữa các mục tiêu chính sách.
""")

# =========================================================
# 7.5. CÂU HỎI THẢO LUẬN CHÍNH SÁCH
# =========================================================

st.header("7.5. Câu hỏi thảo luận chính sách")

# =========================================================
# CÂU A
# =========================================================

st.subheader(
    "a) Khi quan sát đường biên Pareto, em thấy đánh đổi giữa tăng trưởng và bao trùm có rõ ràng không? Mức đánh đổi đó nói lên điều gì về cơ cấu kinh tế Việt Nam?"
)

st.markdown("""
Khi quan sát tập nghiệm Pareto thu được từ NSGA-II,
có thể thấy mối đánh đổi giữa:

- Tăng trưởng GDP
- Bao trùm xã hội

diễn ra khá rõ ràng.

Các nghiệm có:

- GDP Gain rất cao

thường tập trung ngân sách vào:

- Đông Nam Bộ
- Đồng bằng sông Hồng
- Các hạng mục AI và hạ tầng số có hiệu quả kinh tế lớn

Tuy nhiên,
điều này làm gia tăng:

- Chênh lệch phát triển vùng miền
- Mức độ tập trung nguồn lực
- Bất bình đẳng tiếp cận công nghệ

Ngược lại,
các nghiệm ưu tiên:

- Bao trùm xã hội
- Công bằng vùng miền
- Phát triển xanh

thường:

- Phân bổ ngân sách đồng đều hơn
- Tăng đầu tư cho nhân lực số và hạ tầng cơ bản
- Ưu tiên các vùng khó khăn như Tây Nguyên và miền núi phía Bắc

Nhưng khi đó:

- Tốc độ tăng trưởng GDP tối đa sẽ giảm xuống

### Ý nghĩa đối với cơ cấu kinh tế Việt Nam

Kết quả phản ánh rằng:

- Các cực tăng trưởng số của Việt Nam đang tập trung mạnh tại một số vùng phát triển
- Trong khi nhiều địa phương vẫn hạn chế về hạ tầng số và khả năng hấp thụ AI

Do đó,
nếu chỉ tối ưu tăng trưởng,
nguồn lực sẽ tiếp tục dồn về các trung tâm lớn
và làm gia tăng khoảng cách số giữa các vùng.
""")

# =========================================================
# CÂU B
# =========================================================

st.subheader(
    "b) Trọng số (0,40; 0,25; 0,20; 0,15) có phản ánh đúng ưu tiên hiện tại của Việt Nam không? Em sẽ điều chỉnh thế nào để phù hợp với cam kết COP26 và Quyết định 127/QĐ-TTg?"
)

st.markdown("""
Bộ trọng số:

- 0,40 cho tăng trưởng
- 0,25 cho bao trùm
- 0,20 cho môi trường
- 0,15 cho an ninh

nhìn chung phản ánh khá đúng định hướng phát triển hiện nay của Việt Nam theo Văn kiện Đại hội XIII.

Trong giai đoạn hiện nay,
Việt Nam vẫn ưu tiên:

- Tăng trưởng kinh tế
- Chuyển đổi số
- Nâng cao năng lực cạnh tranh quốc gia

Do đó,
việc đặt trọng số cao nhất cho tăng trưởng là hợp lý.

Tuy nhiên,
nếu xét theo:

- Cam kết phát thải ròng bằng 0 tại COP26
- Chiến lược AI quốc gia theo Quyết định 127/QĐ-TTg

thì trọng số môi trường nên được nâng cao hơn.

### Đề xuất điều chỉnh

Có thể điều chỉnh theo hướng:

- Tăng trưởng: 0,35
- Bao trùm: 0,25
- Môi trường: 0,25
- An ninh dữ liệu: 0,15

Hoặc trong dài hạn:

- Tăng trưởng: 0,30
- Bao trùm: 0,25
- Môi trường: 0,30
- An ninh dữ liệu: 0,15

### Ý nghĩa chính sách

Điều này phản ánh xu hướng chuyển từ:

- Tang truong bang moi gia

sang:

- Phát triển xanh
- Kinh tế số bền vững
- AI có trách nhiệm
""")

# =========================================================
# CÂU C
# =========================================================

st.subheader(
    "c) Vai trò của NSGA-II ở đây có gì khác so với LP đơn mục tiêu? Nó có thay thế được quyết định chính trị không?"
)

st.markdown("""
Khác với mô hình LP đơn mục tiêu,
NSGA-II không tìm một nghiệm tối ưu duy nhất,
mà tạo ra cả một tập nghiệm Pareto gồm nhiều phương án chính sách khác nhau.

Trong LP truyền thống:

- Mô hình thường chỉ tối đa hóa một mục tiêu
- Ví dụ GDP hoặc lợi ích kinh tế

Tuy nhiên,
trong thực tế,
các mục tiêu quốc gia thường xung đột với nhau:

- Tăng trưởng kinh tế
- Công bằng xã hội
- Môi trường
- An ninh dữ liệu

khó có thể tối ưu đồng thời.

NSGA-II cho phép:

- Mô phỏng các trade-off
- Đánh giá nhiều kịch bản
- Hỗ trợ lựa chọn phương án phù hợp với ưu tiên chiến lược

### Tuy nhiên, NSGA-II không thể thay thế quyết định chính trị

Thuật toán chỉ là công cụ hỗ trợ định lượng.

Quyết định cuối cùng vẫn cần phụ thuộc vào:

- Ưu tiên phát triển quốc gia
- Yếu tố địa - chính trị
- An ninh quốc gia
- Công bằng xã hội
- Mục tiêu chiến lược dài hạn

Nói cách khác:

- NSGA-II giúp ra quyết định tốt hơn
- Nhưng không thể thay thế vai trò của Nhà nước trong hoạch định chính sách công
""")
st.success("🎉 Hoàn thành bài 7.")
# =========================================================
# OUTPUT CHO BÀI 12 - AIDEOM VN
# =========================================================

st.markdown("---")
st.header("📤 Output cho AIDEOM-VN Dashboard")

# =========================================================
# TOP NGHIỆM PARETO
# =========================================================

top_pareto = pareto_df.sort_values(
    "C_star",
    ascending=False
).head(5)

# =========================================================
# OUTPUT MODULE M7
# =========================================================

M5_OUTPUT_B7 = {

    # =====================================
    # Pareto tốt nhất
    # =====================================
    "best_compromise_solution": {

        "GDP_gain": round(
            best_solution["GDP_gain"], 2
        ),

        "Bao_trum": round(
            best_solution["Bao_trum"], 2
        ),

        "Phat_thai": round(
            best_solution["Phat_thai"], 2
        ),

        "Rui_ro": round(
            best_solution["Rui_ro"], 2
        ),

        "C_star": round(
            best_solution["C_star"], 4
        )
    },
    # =====================================
    # RISK COMPONENTS CHO BÀI 12
    # =====================================
    "risk_components": {

        "cyber_risk": round(
            best_solution["Rui_ro"], 2
        ),

        "environmental_risk": round(
            best_solution["Phat_thai"], 2
        ),

        "dependency_risk": round(
            best_solution["Rui_ro"] * 0.8, 2
        )
    },

    "risk_score": round(
        0.35 * best_solution["Rui_ro"]
        + 0.35 * best_solution["Phat_thai"]
        + 0.30 * best_solution["Rui_ro"] * 0.8,
        2
    ),
    # =====================================
    # GDP max
    # =====================================
    "max_growth_solution": {

        "GDP_gain": round(
            growth_best["GDP_gain"], 2
        ),

        "Bao_trum": round(
            growth_best["Bao_trum"], 2
        ),

        "Phat_thai": round(
            growth_best["Phat_thai"], 2
        ),

        "Rui_ro": round(
            growth_best["Rui_ro"], 2
        )
    },

    # =====================================
    # Trade-off
    # =====================================
    "tradeoff_analysis": {

        "loss_baotrum_percent": round(
            loss_baotrum, 2
        ),

        "loss_environment_percent": round(
            loss_environment, 2
        )
    },

    # =====================================
    # Tổng quan Pareto
    # =====================================
    "pareto_summary": {

        "num_pareto_solutions": int(
            len(pareto_df)
        ),

        "best_C_star": round(
            pareto_df["C_star"].max(), 4
        ),

        "avg_C_star": round(
            pareto_df["C_star"].mean(), 4
        )
    },

    # =====================================
    # TOP 5 nghiệm
    # =====================================
    "top5_pareto": top_pareto.to_dict(
        orient="records"
    )
}

# =========================================================
# HIỂN THỊ
# =========================================================

def get_b7_output():
    return M5_OUTPUT_B7


if __name__ == "__main__":
    st.json(get_b7_output())

st.success("""
✅ M7 Output đã sẵn sàng để tích hợp vào:
Bài 12 — AIDEOM-VN Dashboard
""")