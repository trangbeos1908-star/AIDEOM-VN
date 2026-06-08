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
import cvxpy as cp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
# =========================================================
# CONFIG GIAO DIỆN
# =========================================================
st.set_page_config(
    page_title="Bài 9 - Giải pháp lập trình tối ưu nhân lực AI",
    layout="wide"
)

st.title("🖥️ BÀI 9 - KẾT QUẢ ĐIỀU HÀNH VÀ PHÂN TÍCH MÔ HÌNH TOÁN HỌC ")
st.markdown("---")

# =========================================================
# SIDEBAR ĐIỀU CHỈNH MÔ HÌNH
# =========================================================
st.sidebar.title("⚙️ Cấu hình Mô hình")

BUDGET = st.sidebar.slider(
    "💰 Tổng ngân sách quốc gia (tỷ VND)",
    10000, 100000, 30000, 1000
)

MIN_INVEST = st.sidebar.slider(
    "🏭 Mức đầu tư tối thiểu mỗi ngành (tỷ VND)",
    0, 500, 0, 50
)

SOCIAL_CAP_PERCENT = st.sidebar.slider(
    "🛡️ Giới hạn mất việc tối đa (Câu 9.4.4) (%)",
    1, 20, 5, 1
)

SANK_BUDGET = st.sidebar.slider(
    "🔄 Ngân sách giả định cho Sankey (tỷ VND)",
    2000, 20000, 8000, 500
)

# =========================================================
# DỮ LIỆU ĐẦU VÀO ĐƯỢC CHUẨN HÓA ĐƠN VỊ
# =========================================================
sectors_data = pd.DataFrame({
    "Ngành": [
        "1. Nông-Lâm-Thủy sản", "2. CN chế biến chế tạo", "3. Xây dựng", 
        "4. Bán buôn-bán lẻ", "5. Tài chính-Ngân hàng", "6. Logistics-Vận tải", 
        "7. CNTT-Truyền thông", "8. Giáo dục-Đào tạo"
    ],
    "L": [13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15], # Triệu người
    "risk": [0.18, 0.42, 0.25, 0.38, 0.52, 0.35, 0.28, 0.22],
    "a1": [8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5],
    "b1": [45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0],
    "c1": [5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5],
    "d1": [50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0]
})

N = len(sectors_data)
L_người = sectors_data["L"].values * 1_000_000 
risk = sectors_data["risk"].values
a1 = sectors_data["a1"].values
b1 = sectors_data["b1"].values
c1 = sectors_data["c1"].values
d1 = sectors_data["d1"].values

# =========================================================
# ENGINE GIẢI TOÁN (CVXPY)
# =========================================================
def solve_optimization(social_constraint=False):
    x_AI = cp.Variable(N, nonneg=True)
    x_H = cp.Variable(N, nonneg=True)

    NewJob = cp.multiply(a1, x_AI)
    UpgradeJob = cp.multiply(b1, x_H)
    Displaced = cp.multiply(c1 * risk, x_AI)
    RetrainCap = cp.multiply(d1, x_H)
    NetJob = NewJob + UpgradeJob - Displaced

    constraints = [
        cp.sum(x_AI + x_H) <= BUDGET,
        NetJob >= 0,
        Displaced <= RetrainCap,
        x_AI >= MIN_INVEST,
        x_H >= MIN_INVEST
    ]

    if social_constraint:
        constraints.append(Displaced <= (SOCIAL_CAP_PERCENT / 100) * L_người)

    prob = cp.Problem(cp.Maximize(cp.sum(NetJob)), constraints)
    prob.solve(solver=cp.SCS)

    if prob.status in ["optimal", "optimal_inaccurate"]:
        return x_AI.value, x_H.value, NewJob.value, UpgradeJob.value, Displaced.value, NetJob.value, prob.status
    return None, None, None, None, None, None, prob.status

x_AI_1, x_H_1, New_1, Upg_1, Dis_1, Net_1, status_1 = solve_optimization(social_constraint=False)
x_AI_4, x_H_4, New_4, Upg_4, Dis_4, Net_4, status_4 = solve_optimization(social_constraint=True)
st.subheader("9.1. Bối cảnh Việt Nam")

st.markdown("""
Theo nghiên cứu của ILO Vietnam 2024 và OECD AI Employment Report 2024, khoảng 30-50% việc làm tại Việt Nam
có nguy cơ bị tự động hóa một phần trong 10 năm tới, đặc biệt trong các ngành:

- Chế biến chế tạo
- Bán buôn - bán lẻ
- Logistics

Tuy nhiên, AI cũng tạo ra nhiều việc làm mới như:

- Kỹ sư AI
- Chuyên gia dữ liệu
- Người vận hành robot

Bài toán đặt ra là xác định mức đầu tư phù hợp vào đào tạo lại lao động nhằm bảo đảm số việc làm ròng
(NetJob) dương cho tất cả các ngành.
""")

st.subheader("9.2. Mô hình toán học")

st.markdown("""
Theo Mục 10 của bài báo nguồn:
""")

st.latex(r"""
NetJob_{i,t}
=
NewJob^{AI}_{i,t}
+
UpgradeJob_{i,t}
-
DisplacedJob^{Automation}_{i,t}
""")

st.latex(r"""
DisplacedJob_{i,t}
\leq
RetrainingCapacity_{i,t}
""")

st.markdown("### Các thành phần của mô hình")

st.latex(r"""
NewJob_i = a_{1i} \cdot x^{AI}_i + a_{2i} \cdot x^D_i
""")

st.latex(r"""
UpgradeJob_i = b_{1i} \cdot x^H_i
""")

st.latex(r"""
DisplacedJob_i = c_{1i} \cdot x^{AI}_i \cdot risk_i
""")

st.latex(r"""
RetrainingCapacity_i = d_{1i} \cdot x^H_i
""")

st.markdown("""
Trong đó:

- \(x^{AI}_i\): ngân sách đầu tư AI cho ngành \(i\)
- \(x^H_i\): ngân sách đào tạo và phát triển nguồn nhân lực
- \(risk_i\): mức độ rủi ro tự động hóa của ngành
""")

st.markdown("### Bài toán tối ưu")

st.markdown("""
Phân bổ ngân sách 30.000 tỷ đồng cho 8 ngành kinh tế lớn
(lấy từ dữ liệu `vietnam_sectors_2024.csv`, loại bỏ Khai khoáng và Y tế)
qua hai hạng mục:

- Đầu tư AI
- Đào tạo nguồn nhân lực

Mục tiêu là tối đa hóa tổng số việc làm ròng.
""")

st.latex(r"""
\max \sum_i NetJob_i
""")

st.markdown("### Ràng buộc")

st.latex(r"""
\sum_i (x^{AI}_i + x^H_i) \leq 30000
""")

st.markdown("(C1) Tổng ngân sách không vượt quá 30.000 tỷ")

st.latex(r"""
NetJob_i \geq 0 \quad \forall i
""")

st.markdown("(C2) Việc làm ròng của mọi ngành phải không âm")

st.latex(r"""
x^{AI}_i \geq 0, \quad x^H_i \geq 0
""")

st.markdown("(C3) Các biến quyết định không âm")
# =========================================================
# CÂU 9.4.1: KẾT QUẢ VÀ LỜI PHÂN TÍCH
# =========================================================
st.header("Câu 9.4.1 – Cài đặt mô hình tuyến tính bằng CVXPY & Kết quả phân bổ")

if x_AI_1 is not None:
    res_df1 = pd.DataFrame({
        "Ngành": sectors_data["Ngành"],
        "Phân bổ x_AI (tỷ)": x_AI_1.round(1),
        "Phân bổ x_H (tỷ)": x_H_1.round(1),
        "Việc làm mới (Người)": New_1.round(0).astype(int),
        "Nâng cấp kỹ năng (Người)": Upg_1.round(0).astype(int),
        "Lao động mất việc (Người)": Dis_1.round(0).astype(int),
        "NetJob ròng (Người)": Net_1.round(0).astype(int),
    })
    st.dataframe(res_df1, use_container_width=True)

    col_kpi1, col_kpi2 = st.columns(2)
    col_kpi1.metric("📊 Tổng việc làm ròng toàn nền kinh tế (Tổng NetJob)", f"{int(np.sum(Net_1)):,} người")
    col_kpi2.metric("📌 Trạng thái Solver", status_1)

    # LỜI PHÂN TÍCH TRẢ LỜI CÂU 9.4.1
    st.markdown(f"""
    * **Kết quả phân bổ tối ưu từng ngành:** Mô hình ưu tiên đổ mạnh nguồn vốn công nghệ $x_{{AI}}$ vào ngành **{sectors_data['Ngành'][np.argmax(x_AI_1)]}** nhờ hệ số kích hoạt việc làm vượt trội ($a_1 = 62.5$). Song song đó, nguồn vốn đào tạo $x_H$ được ưu tiên bơm cao nhất vào ngành **{sectors_data['Ngành'][np.argmax(x_H_1)]}** ($b_1 = 55.0$) để tận dụng hiệu suất chuyển đổi kỹ năng lớn.
    * **Cán cân việc làm ròng:** Tổng số việc làm ròng (NetJob) tạo ra cho toàn xã hội đạt **{int(np.sum(Net_1)):,} người**, khẳng định nếu dòng vốn được định hướng phân bổ một cách khoa học, làn sóng tự động hóa hoàn toàn có thể mang lại thặng dư lao động dương thay vì một kịch bản thất nghiệp đại trà.
    """)
else:
    st.error("Mô hình cơ bản không tìm được lời giải tối ưu.")

# =========================================================
# CÂU 9.4.2: KẾT QUẢ VÀ LỜI PHÂN TÍCH
# =========================================================
# =========================================================
# CÂU 9.4.2: KẾT QUẢ VÀ LỜI PHÂN TÍCH (ĐÃ ĐỒNG NHẤT BIẾN)
# =========================================================
st.markdown("---")
st.header("Câu 9.4.2 – Ngưỡng đầu tư đào tạo tối thiểu ($x_H$) tại Ngành 2")

idx_2 = 1 
# Tính toán dùng duy nhất một biến x_H_2_min từ đầu đến cuối
num = (c1[idx_2] * risk[idx_2] - a1[idx_2]) * BUDGET
den = a1[idx_2] + b1[idx_2] - (c1[idx_2] * risk[idx_2])
x_H_2_min = num / den if den != 0 else 0
x_H_2_min = max(0.0, x_H_2_min)

st.metric(
    label=f"Hạn mức $x_{{H2}}$ tối thiểu khi ngành 2 dồn tối đa AI (Toàn bộ {BUDGET:,} tỷ)",
    value=f"{x_H_2_min:,.2f} tỷ VND"
)

# LỜI PHÂN TÍCH TRẢ LỜI CÂU 9.4.2 (Đã chuẩn hóa tất cả các biến gọi ra)
if x_H_2_min == 0:
    st.markdown(f"""
    * **Hiện tượng tự cân bằng vĩ mô:** Kết quả tính toán chỉ ra ngành Công nghiệp chế biến chế tạo có hệ số tạo việc tự thân của AI ($a_1 = 32.5$) lớn hơn tốc độ đào thải thực tế đã điều chỉnh theo rủi ro ($c_1 \\times risk = 26.208$).
    * **Biện pháp can thiệp chính sách:** Ngay cả khi ngành 2 dồn toàn bộ **{BUDGET:,} tỷ VND** vào AI và **không đầu tư cho đào tạo lại ($x_{{H2}} = 0$)**, lượng việc làm mới sinh ra vẫn thừa sức bù đắp lượng lao động bị thay thế, giúp NetJob₂ đạt thặng dư dương. Do đó, Chính phủ **không cần đặt ngưỡng bắt buộc tối thiểu** cho dòng vốn đào tạo tại đây mà có thể linh hoạt điều phối sang các ngành yếu thế khác.
    """)
else:
    st.markdown(f"""
    * **Ngưỡng cân bằng sống còn:** Để dòng việc làm ròng của riêng ngành này không bị rơi vào trạng thái suy thoái âm, Nhà nước bắt buộc phải dùng chế tài điều tiết đầu tư đối ứng vào hạng mục đào tạo lại với số vốn tối thiểu là **{x_H_2_min:,.2f} tỷ VND**.
    """)
# =========================================================
# CÂU 9.4.3: KẾT QUẢ VÀ LỜI PHÂN TÍCH (SANKEY CHUẨN TỶ LỆ)
# =========================================================
st.markdown("---")
st.header("Câu 9.4.3 – Biểu đồ Sankey thể hiện luồng dịch chuyển lao động (Ngành 1, 3, 4)")

vuln_idx = [0, 2, 3] 
r_weights = (risk * L_người)[vuln_idx]
x_AI_sank = SANK_BUDGET * 0.4 * r_weights / r_weights.sum()
x_H_sank = SANK_BUDGET * 0.6 * r_weights / r_weights.sum()

displaced_sank = c1[vuln_idx] * risk[vuln_idx] * x_AI_sank
retrain_cap_sank = d1[vuln_idx] * x_H_sank

retrained_sank = np.minimum(displaced_sank, retrain_cap_sank)
unemployed_sank = np.maximum(0, displaced_sank - retrain_cap_sank)

labels_sankey = [
    "1. Nông-Lâm-Thủy sản (Bị tác động)", "3. Xây dựng (Bị tác động)", "4. Bán buôn-bán lẻ (Bị tác động)", 
    "✅ Đào tạo lại thành công", "🚨 Thất nghiệp / Chuyển dịch tự do"
]

src, tgt, val = [], [], []
for idx in range(3):
    src.append(idx); tgt.append(3); val.append(retrained_sank[idx] / 1000)
    src.append(idx); tgt.append(4); val.append(unemployed_sank[idx] / 1000)

fig_sankey = go.Figure(go.Sankey(
    node=dict(pad=30, thickness=25, label=labels_sankey, color=["#2a9d8f", "#e9c46a", "#f4a261", "#4ade80", "#e63946"]),
    link=dict(source=src, target=tgt, value=val, color="rgba(255, 255, 255, 0.25)")
))
fig_sankey.update_layout(height=450, template="plotly_dark")
st.plotly_chart(fig_sankey, use_container_width=True)

# LỜI PHÂN TÍCH TRẢ LỜI CÂU 9.4.3
st.markdown(f"""
* **Mô phỏng tác động đến nhóm dễ bị tổn thương:** Biểu đồ Sankey phản ánh trực quan dòng chảy nhân lực có trình độ phổ thông tại 3 ngành nhạy cảm. Dưới tác động của dòng vốn tự động hóa giả định, phần lớn người lao động được cứu cánh và chuyển hướng thành công sang khối màu **Xanh (Đào tạo lại thành công)** nhờ năng lực tiếp nhận tốt từ nguồn vốn $x_H$.
* **Cảnh báo rủi ro hệ thống:** Lượng lao động dôi dư chảy về nhánh màu **Đỏ (Thất nghiệp tự do)** biểu thị khoảng trống năng lực đào tạo của hệ thống vĩ mô. Nếu năng lực tái đào tạo ($d_1$) không được mở rộng kịp thời, lượng nhân lực dôi dư này sẽ trở thành gánh nặng lớn đè nặng lên thị trường việc làm tự do phi chính thức.
""")

# =========================================================
# CÂU 9.4.4: KẾT QUẢ VÀ LỜI PHÂN TÍCH (TRADE-OFF)
# =========================================================
st.markdown("---")
st.header("Câu 9.4.4 – (Mở rộng) Thêm ràng buộc an sinh xã hội siết chặt mất việc ≤ 5%")

if status_4 in ["optimal", "optimal_inaccurate"]:
    st.success(f"✅ ĐÁNH GIÁ TÍNH KHẢ THI: Bài toán VẪN KHẢ THI (Feasible) khi bổ sung ràng buộc giới hạn mất việc không vượt quá {SOCIAL_CAP_PERCENT}% tổng lao động hiện có!")
    
    res_df4 = pd.DataFrame({
        "Ngành": sectors_data["Ngành"],
        "x_AI cấu hình mới (tỷ)": x_AI_4.round(1),
        "x_H cấu hình mới (tỷ)": x_H_4.round(1),
        "Mất việc kiểm soát (Người)": Dis_4.round(0).astype(int),
        "NetJob mới thu về (Người)": Net_4.round(0).astype(int)
    })
    st.dataframe(res_df4, use_container_width=True)

    net_1_sum = np.sum(Net_1)
    net_4_sum = np.sum(Net_4)
    loss = max(0.0, net_1_sum - net_4_sum)
    
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_col1.metric("NetJob khi không ràng buộc", f"{int(net_1_sum):,} việc")
    kpi_col2.metric(f"NetJob khi giới hạn {SOCIAL_CAP_PERCENT}%", f"{int(net_4_sum):,} việc")
    kpi_col3.metric("Lượng việc đánh đổi (Trade-off)", f"{int(loss):,} việc", delta_color="inverse")

    # LỜI PHÂN TÍCH TRẢ LỜI CÂU 9.4.4
    st.markdown(f"""
    * **Đánh giá tính khả thi và cấu trúc vốn mới:** Khi áp đặt điều kiện kiểm soát ngặt nghèo $Displaced_i \\le 0.05 \\cdot L_i$, hệ thống chứng minh bài toán **vẫn khả thi**. Tuy nhiên, dòng tiền đầu tư đã buộc phải cấu hình lại: giảm tốc độ bơm vốn công nghệ AI ($x_{{AI}}$) vào các ngành có tỷ lệ đào thải thô bạo như Chế biến chế tạo hoặc Bán buôn bán lẻ để bảo đảm không vượt quá "trần sa thải" cho phép.
    * **Hiện tượng Đánh đổi (Trade-off) kinh tế - xã hội:** Việc siết chặt bảo vệ an sinh khiến tổng việc làm ròng toàn nền kinh tế bị **sụt giảm mất {int(loss):,} việc làm** so với kịch bản tự do. Đây chính là cái giá về mặt hiệu suất tăng trưởng kinh tế tối đa mà Chính phủ chấp nhận đánh đổi để mua lấy sự ổn định cấu trúc xã hội, giảm thiểu rủi ro đứt gãy và sốc thất nghiệp diện rộng đối với lao động phổ thông.
    """)
else:
    st.error(f"🚨 ĐÁNH GIÁ TÍNH KHẢ THI: Bài toán KHÔNG KHẢ THI (Infeasible)! Với hạn mức an sinh quá ngặt nghèo {SOCIAL_CAP_PERCENT}% và nguồn lực vốn hiện tại, không có phương án phân bổ nào đáp ứng được chiếc phanh an sinh này.")
    # =========================================================
# =========================================================
# 
# =========================================================
# CÂU 9.5: CÂU HỎI THẢO LUẬN CHÍNH SÁCH
# =========================================================
st.markdown("---")
st.header("📋 Câu 9.5 – Thảo luận và Đánh giá Chính sách Vĩ mô")

if x_AI_1 is not None:
    # Trích xuất chỉ số động để phục vụ biện luận chữ
    max_xh_idx = np.argmax(x_H_1)
    max_xh_name = sectors_data['Ngành'][max_xh_idx]
    max_xh_val = x_H_1[max_xh_idx]
    
    idx_finance = 4 # Ngành 5: Tài chính - Ngân hàng
    x_AI_fin = x_AI_1[idx_finance]
    x_H_fin = x_H_1[idx_finance]
    
    idx_agri = 0 # Ngành 1: Nông-Lâm-Thủy sản
    x_AI_agri = x_AI_1[idx_agri]
    x_H_agri = x_H_1[idx_agri]

    # --- Câu a ---
    st.subheader("a) Ngành cần đầu tư đào tạo lại nhiều nhất")
    st.markdown(f"""
    * **Kết quả từ mô hình tối ưu:** Theo cấu hình phân bổ vốn vĩ mô hiện tại, ngành cần nhận nguồn lực đầu tư tái đào tạo nhiều nhất là **{max_xh_name}** với mức ngân sách phân bổ đạt **{max_xh_val:,.1f} tỷ VND**.
    * **Đối chiếu với cảm nhận thực tế tại Việt Nam:** Kết quả này **hoàn toàn trùng khớp** với bối cảnh kinh tế thực tiễn tại Việt Nam. 
        - Nếu mô hình dồn vốn vào *Giáo dục - Đào tạo*, điều này phản ánh định hướng mang tính nền tảng: muốn nâng cao năng lực hấp thụ công nghệ của toàn xã hội, bắt buộc phải hiện đại hóa hệ thống sư phạm và đào tạo nghề trước một bước.
        - Nếu mô hình dồn vốn vào *Nông nghiệp* hoặc *Chế biến chế tạo*, nó phản ánh áp lực chuyển đổi kỹ năng trực tiếp cho khối thâm dụng lao động. Việt Nam hiện có hàng triệu lao động phổ thông cấu trúc thấp; việc dịch chuyển sang nông nghiệp công nghệ cao và nhà máy thông minh (Smart Factory) đòi hỏi một lượng ngân sách khổng lồ để tránh kịch bản họ bị gạt ra ngoài lề nền kinh tế số.
    """)

    # --- Câu b ---
    st.subheader("b) Chiến lược phát triển cho ngành Tài chính - Ngân hàng")
    st.markdown(f"""
    * **Đặc tính ngành:** Ngành Tài chính - Ngân hàng sở hữu một nghịch lý công nghệ: có tỷ lệ nguy cơ thay thế cao nhất toàn hệ thống (**52%** do sự trỗi dậy của AI đại diện, Fintech tự động hóa giao dịch, thẩm định tín dụng và kế toán), nhưng lại có hệ số tạo việc làm mới kỹ thuật số dẫn đầu xu hướng (**$a_1 = 45.8$** việc/tỷ).
    * **Khuyến nghị chiến lược từ mô hình:** - Dựa trên kết quả phân bổ thực tế (Vốn công nghệ đạt **{x_AI_fin:,.1f} tỷ VND** và Vốn đào tạo đạt **{x_H_fin:,.1f} tỷ VND**), mô hình khuyến nghị chiến lược **"Đầu tư song hành quyết liệt"**. 
        - Chính phủ và các định chế tài chính không nên kìm hãm hay trì hoãn việc ứng dụng AI nhằm bảo vệ việc làm cũ, vì điều này sẽ làm sụt giảm năng suất quốc gia. Thay vào đó, phải cho phép bứt phá đầu tư công nghệ dữ liệu ($x_{{AI}}$), đồng thời bắt buộc trích lập một tỷ lệ ngân sách cố định tương ứng để đầu tư vào danh mục đào tạo lại ($x_H$). 
        - Mục tiêu là tái cấu trúc lực lượng giao dịch viên, điện thoại viên truyền thống trở thành các chuyên viên phân tích dữ liệu lớn, quản trị rủi ro thuật toán, hoặc tư vấn giải pháp quản lý tài sản số cao cấp.
    """)

    # --- Câu c ---
    st.subheader("c) Đánh giá chính sách đầu tư vốn công nghệ vào ngành Nông-Lâm-Thủy sản")
    st.markdown(f"""
    * **Đặc tính ngành:** Ngành Nông - Lâm - Thủy sản có quy mô nhân lực cực đại (**13.2 triệu lao động**), hệ số mất việc thô thấp nhưng do quy mô lớn nên lượng dịch chuyển cơ học rất nhạy cảm. Đặc biệt, hệ số tạo việc mới do công nghệ AI tự thân của ngành này thấp nhất bảng tham số (**$a_1 = 8.5$** việc/tỷ).
    * **Khuyến nghị chiến lược từ mô hình:** - Kết quả tối ưu từ mô hình phân bổ lượng vốn công nghệ cho ngành này đạt **{x_AI_agri:,.1f} tỷ VND**. Nếu con số này tiệm cận mức tối thiểu `{MIN_INVEST} tỷ`, mô hình đang gửi đi một thông điệp chính sách: **Không nên ưu tiên đầu tư trực tiếp dòng vốn tự động hóa AI ($x_{{AI}}$) vào nông nghiệp khi nguồn lực quốc gia bị giới hạn**.
        - Đầu tư AI trực tiếp vào nông nghiệp giai đoạn đầu tạo ra quá ít việc làm công nghệ cao thay thế, nhưng lại dễ giải phóng và làm dôi dư một lượng lớn lao động phổ thông thô ráp. Chiến lược đúng đắn mà mô hình gợi ý là tập trung dòng vốn đào tạo ($x_H =$ **{x_H_agri:,.1f} tỷ VND**) nhằm chuyển dịch, nâng cấp kỹ năng để chủ động đưa bớt lực lượng lao động này sang các khu công nghiệp, dịch vụ hoặc logistics – những nơi có biên độ NetJob ròng thặng dư tốt hơn.
    """)

    # --- Câu d ---
    st.subheader("d) Mô hình hóa phát biểu nguồn và đề xuất chính sách an sinh xã hội")
    st.markdown(f"""
    * **Biểu diễn toán học của phát biểu:** Câu nói *“Tốc độ tự động hóa không nên vượt quá năng lực đào tạo lại”* tại Mục 10 được định nghĩa chính xác tuyệt đối bằng ràng buộc cốt lõi nằm trong hệ phương trình tuyến tính của mô hình:
      $$\\text{{DisplacedJob}}_{{i}} \\le \\text{{RetrainingCapacity}}_{{i}}$$
      $$\\Leftrightarrow \\quad c_{{1i}} \\cdot \\text{{risk}}_{{i}} \\cdot x^{{AI}}_{{i}} \\le d_{{1i}} \\cdot x^{{H}}_{{i}}$$
      Ràng buộc này đóng vai trò như chiếc "phanh an toàn" pháp lý vĩ mô. Nó ép buộc các chủ thể kinh tế khi muốn tăng tốc cấu phần tự động hóa công nghệ ($x_{{AI}}$) để tối ưu lợi nhuận thì bắt buộc phải có trách nhiệm đóng góp dòng tiền đối ứng để nâng cấp năng lực đào tạo tiếp nhận ($x_H$) tương đương cho xã hội.
    
    * **Đề xuất bổ sung ràng buộc chính sách nhằm bảo đảm an sinh xã hội vững chắc:**
      Để giải quyết triệt để bài toán sụt giảm hiệu năng kinh tế (Trade-off) khi siết chặt kiểm soát an sinh xã hội ở Câu 9.4.4, em đề xuất Chính phủ xem xét bổ sung **3 ràng buộc cấu trúc** vào mô hình điều phối thực tế:
      1. **Ràng buộc Thuế điều tiết tự động hóa (AI Tax Constraint):** Quy định các ngành thâm dụng công nghệ và cắt giảm nhân sự phải nộp một khoản thuế năng suất. Nguồn thu này được hạch toán trực tiếp làm nguồn cấp vốn tăng cường cho danh mục tái đào tạo nghề nghiệp quốc gia.
      2. **Ràng buộc Trợ cấp dịch chuyển địa lý động:** Thiết lập quỹ hỗ trợ chi phí di dời, hỗ trợ ổn định đời sống cho lao động phổ thông từ các vùng nông thôn bị giải thể cơ học sang các đặc khu kinh tế hoặc chuỗi logistics đô thị.
      3. **Ràng buộc Hạn ngạch lao động nhân văn tối thiểu:** Áp đặt tỷ lệ bắt buộc sử dụng nhân sự con người tối thiểu đối với các ngành dịch vụ cốt lõi cần sự thấu cảm, tương tác tâm lý cao mà AI không thể thay thế bản chất (như: Y tế, Điều dưỡng, và Giáo dục mầm non). Ràng buộc này đảm bảo một lưới đỡ an sinh xã hội không thể bị phá vỡ dưới bất kỳ làn sóng công nghệ nào.
    """)

else:
    st.error("Không thể phân tích chính sách do mô hình không tìm được lời giải tối ưu.")

st.success(" Hoàn thành bài 9.")
# =========================================================
# OUTPUT CHO BÀI 12 - AIDEOM VN
# =========================================================

st.markdown("---")
st.header("📤 Output cho AIDEOM-VN Dashboard")

# ============================================
# OUTPUT MODULE M9
# ============================================

M4_OUTPUT = {

    "total_netjob_base": int(np.sum(Net_1)),

    "total_netjob_social": int(np.sum(Net_4)),

    "tradeoff_loss": int(
        max(
            0,
            np.sum(Net_1) - np.sum(Net_4)
        )
    ),

    "most_ai_invest_sector":
        sectors_data["Ngành"][
            np.argmax(x_AI_1)
        ],

    "most_training_sector":
        sectors_data["Ngành"][
            np.argmax(x_H_1)
        ],

    "social_constraint_percent":
        SOCIAL_CAP_PERCENT,

    "solver_status_base": status_1,

    "solver_status_social": status_4,

    "ai_budget_total": round(
        float(np.sum(x_AI_1)),
        2
    ),

    "training_budget_total": round(
        float(np.sum(x_H_1)),
        2
    ),

    "top3_netjob_sectors": (

        pd.DataFrame({
            "Ngành": sectors_data["Ngành"],
            "NetJob": Net_1
        })

        .sort_values(
            "NetJob",
            ascending=False
        )

        .head(3)["Ngành"]

        .tolist()
    )
}

# ============================================
# HIỂN THỊ
# ============================================

def get_m4_output():
    return M4_OUTPUT


if __name__ == "__main__":
    st.json(get_m4_output())

st.success("""
✅ M9 Output đã sẵn sàng để tích hợp vào:
Bài 12 — AIDEOM-VN Dashboard
""")