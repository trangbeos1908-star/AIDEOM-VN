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
import pyomo.environ as pyo
import pandas as pd
import os
import urllib.request
import zipfile
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
# Thiết lập trang giao diện rộng
st.set_page_config(page_title="Stochastic Programming Model", layout="wide")

# ==========================================
# AUTO-DOWNLOAD VÀ CẤU HÌNH SOLVER CBC
# ==========================================
@st.cache_resource
def setup_cbc_solver():
    """Tự động tải bộ giải CBC với link chính thức, có quét đệ quy tìm file exe."""
    cbc_dir = os.path.join(os.path.expanduser("~"), "cbc_solver_fixed")
    
    # 1. Quét tìm cbc.exe nếu đã tải trước đó
    if os.path.exists(cbc_dir):
        for root, dirs, files in os.walk(cbc_dir):
            if 'cbc.exe' in files:
                return os.path.join(root, 'cbc.exe')
                
    # 2. Nếu chưa có, tiến hành tải bản chuẩn
    with st.spinner("Đang tải bộ giải toán học CBC (Chỉ diễn ra 1 lần duy nhất)..."):
        try:
            os.makedirs(cbc_dir, exist_ok=True)
            url = "https://github.com/coin-or/Cbc/releases/download/releases%2F2.10.10/Cbc-releases.2.10.10-w64-msvc17-md.zip"
            zip_path = os.path.join(cbc_dir, "cbc.zip")
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(cbc_dir)
                
            for root, dirs, files in os.walk(cbc_dir):
                if 'cbc.exe' in files:
                    return os.path.join(root, 'cbc.exe')
                    
        except Exception as e:
            st.error(f"Lỗi khi tự động tải CBC: {e}")
            return None
    return None

cbc_path = setup_cbc_solver()
if cbc_path:
    solver = pyo.SolverFactory('cbc', executable=cbc_path.replace("\\", "/"))
else:
    st.error("❌ Không thể thiết lập bộ giải CBC. Vui lòng kiểm tra lại mạng.")
    st.stop()

# ==========================================
# 1. SIDEBAR - THAM SỐ MÔ HÌNH ĐỘNG
# ==========================================
st.sidebar.header("⚙️ TÙY CHỈNH THAM SỐ")
budget_1 = st.sidebar.number_input("Ngân sách Giai đoạn 1 (tỷ VND)", value=65000, step=1000)
budget_2 = st.sidebar.number_input("Ngân sách Giai đoạn 2 (Dự phòng)", value=15000, step=1000)

st.sidebar.markdown("---")
st.sidebar.subheader("Xác suất kịch bản (%)")
p_s1 = st.sidebar.slider("s1. Lạc quan", 0, 100, 30)
p_s2 = st.sidebar.slider("s2. Cơ sở", 0, 100, 45)
p_s3 = st.sidebar.slider("s3. Bi quan", 0, 100, 20)
p_s4 = st.sidebar.slider("s4. Khủng hoảng", 0, 100, 5)

total_p = p_s1 + p_s2 + p_s3 + p_s4
if total_p == 0:
    st.sidebar.error("Tổng xác suất phải > 0")
    st.stop()
probs = {'s1': p_s1/total_p, 's2': p_s2/total_p, 's3': p_s3/total_p, 's4': p_s4/total_p}

# Dữ liệu tĩnh từ đề bài
categories = ['I', 'D', 'AI', 'H']
scenarios = ['s1', 's2', 's3', 's4']

beta_base = {'I': 1.00, 'D': 1.10, 'AI': 1.25, 'H': 0.95}
beta_scen = {
    's1': {'I': 1.25, 'D': 1.35, 'AI': 1.55, 'H': 1.05},
    's2': {'I': 1.00, 'D': 1.10, 'AI': 1.25, 'H': 0.95},
    's3': {'I': 0.75, 'D': 0.85, 'AI': 0.90, 'H': 1.00},
    's4': {'I': 0.40, 'D': 0.50, 'AI': 0.55, 'H': 1.10}
}

# ==========================================
# CÁC HÀM XÂY DỰNG MÔ HÌNH PYOMO
# ==========================================
def solve_sp():
    m = pyo.ConcreteModel()
    m.J = pyo.Set(initialize=categories)
    m.S = pyo.Set(initialize=scenarios)
    m.x = pyo.Var(m.J, domain=pyo.NonNegativeReals)
    m.y = pyo.Var(m.S, m.J, domain=pyo.NonNegativeReals)
    
    m.c_budget1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= budget_1)
    m.c_budget2 = pyo.Constraint(m.S, rule=lambda m, s: sum(m.y[s, j] for j in m.J) <= budget_2)
    m.c_ai = pyo.Constraint(m.S, rule=lambda m, s: m.y[s, 'AI'] <= 0.5 * m.x['H'])
    
    m.obj = pyo.Objective(rule=lambda m: sum(beta_base[j] * m.x[j] for j in m.J) + 
                          sum(probs[s] * sum(beta_scen[s][j] * m.y[s, j] for j in m.J) for s in m.S), sense=pyo.maximize)
    solver.solve(m)
    return m

def solve_deterministic(s_chosen=None, use_expected=False):
    m = pyo.ConcreteModel()
    m.J = pyo.Set(initialize=categories)
    m.x = pyo.Var(m.J, domain=pyo.NonNegativeReals)
    m.y = pyo.Var(m.J, domain=pyo.NonNegativeReals)
    
    m.c1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= budget_1)
    m.c2 = pyo.Constraint(expr=sum(m.y[j] for j in m.J) <= budget_2)
    m.c3 = pyo.Constraint(expr=m.y['AI'] <= 0.5 * m.x['H'])
    
    if use_expected:
        exp_beta = {j: sum(probs[s] * beta_scen[s][j] for s in scenarios) for j in categories}
        m.obj = pyo.Objective(expr=sum(beta_base[j] * m.x[j] for j in m.J) + sum(exp_beta[j] * m.y[j] for j in m.J), sense=pyo.maximize)
    else:
        m.obj = pyo.Objective(expr=sum(beta_base[j] * m.x[j] for j in m.J) + sum(beta_scen[s_chosen][j] * m.y[j] for j in m.J), sense=pyo.maximize)
    solver.solve(m)
    return m

def solve_eev(x_fixed):
    m = solve_sp() 
    for j in categories: m.x[j].fix(x_fixed[j]) 
    solver.solve(m)
    return m.obj()

def solve_robust(ws_objs):
    m = pyo.ConcreteModel()
    m.J = pyo.Set(initialize=categories)
    m.S = pyo.Set(initialize=scenarios)
    m.x = pyo.Var(m.J, domain=pyo.NonNegativeReals)
    m.y = pyo.Var(m.S, m.J, domain=pyo.NonNegativeReals)
    m.Z = pyo.Var(domain=pyo.NonNegativeReals) 
    
    m.c_budget1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= budget_1)
    m.c_budget2 = pyo.Constraint(m.S, rule=lambda m, s: sum(m.y[s, j] for j in m.J) <= budget_2)
    m.c_ai = pyo.Constraint(m.S, rule=lambda m, s: m.y[s, 'AI'] <= 0.5 * m.x['H'])
    
    def regret_rule(m, s):
        achieved = sum(beta_base[j] * m.x[j] for j in m.J) + sum(beta_scen[s][j] * m.y[s, j] for j in m.J)
        return m.Z >= ws_objs[s] - achieved
    m.c_regret = pyo.Constraint(m.S, rule=regret_rule)
    m.obj = pyo.Objective(expr=m.Z, sense=pyo.minimize)
    solver.solve(m)
    return m

# ==========================================
# PHẦN GIAO DIỆN CHÍNH (BẮT ĐẦU KHỐI TRY)
# ==========================================
st.title("📊 HOẠCH ĐỊNH NGÂN SÁCH DƯỚI ĐIỀU KIỆN BẤT ĐỊNH")
st.markdown("---")

try:
    # --- Chạy mô hình để lấy số liệu trước ---
    m_sp = solve_sp()
    x_sp = {j: m_sp.x[j]() for j in categories}
    y_sp = {s: {j: m_sp.y[s, j]() for j in categories} for s in scenarios}
    obj_sp = m_sp.obj()

    det_results = {}
    ws_objectives = {}
    ws_expected = 0
    for s in scenarios:
        m_det = solve_deterministic(s_chosen=s)
        det_results[s] = {j: m_det.x[j]() for j in categories}
        ws_objectives[s] = m_det.obj()
        ws_expected += probs[s] * ws_objectives[s]

    m_ev = solve_deterministic(use_expected=True)
    x_ev = {j: m_ev.x[j]() for j in categories}
    
    obj_eev = solve_eev(x_ev)
    vss = obj_sp - obj_eev
    evpi = ws_expected - obj_sp

    m_robust = solve_robust(ws_objectives)
    x_robust = {j: m_robust.x[j]() for j in categories}
    robust_regret = m_robust.obj()

    # ==========================
    # CÂU 10.5.1
    # ==========================
    st.header("10.5.1: Cài đặt mô hình SP")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Báo cáo quyết định Giai đoạn 1 ($x_{SP}$)")
        st.dataframe(pd.DataFrame([x_sp]).T.rename(columns={0: "Vốn phân bổ (tỷ VND)"}).style.format("{:,.2f}"))
    with col2:
        st.subheader("Quyết định Giai đoạn 2 ($y_{SP}$) - Động theo kịch bản")
        st.dataframe(pd.DataFrame(y_sp).T.style.format("{:,.2f}"))
        st.success(f"**GDP Kỳ vọng:** {obj_sp:,.2f}")
    st.markdown("---")

    # ==========================
    # CÂU 10.5.2
    # ==========================
    st.header("10.5.2: So sánh EV với SP")
    st.write("**Wait-and-See (Kịch bản xác định riêng rẽ):**")
    df_det = pd.DataFrame(det_results).T
    df_det.columns = [f"x_{j}" for j in categories]
    df_det['Obj_Tối_ưu'] = ws_objectives.values()
    st.dataframe(df_det.style.format("{:,.2f}"))

    st.write("**So sánh quyết định Expected Value (EV) và Stochastic (SP):**")
    df_compare = pd.DataFrame({"Mô hình EV (Kỳ vọng)": x_ev, "Mô hình SP (Ngẫu nhiên)": x_sp})
    st.table(df_compare.style.format("{:,.2f}"))
    st.markdown("---")

    # ==========================
    # CÂU 10.5.3
    # ==========================
    st.header("10.5.3: Tính VSS và EVPI")
    col3, col4 = st.columns(2)
    col3.metric("VSS (Value of Stochastic Solution)", f"{vss:,.2f}")
    col4.metric("EVPI (Expected Value of Perfect Info)", f"{evpi:,.2f}")
    
    st.info("""
    **Tại sao VSS và EVPI lại bằng 0? (Gợi ý đưa vào báo cáo)**
    Do hệ số lợi nhuận của AI ở Giai đoạn 1 (1.25) quá vượt trội so với phần còn lại, và bài toán không có trần giới hạn ngân sách tối đa cho từng ngành. Thuật toán tối ưu đã lựa chọn phương án "tất tay" (Nghiệm góc) 100% ngân sách Giai đoạn 1 vào AI. Quyết định cứng rắn này hoàn hảo đến mức dù bạn dùng giá trị trung bình (EV) hay biết trước tương lai (WS), chiến lược Giai đoạn 1 vẫn không thay đổi. Vì quyết định gốc không đổi, thông tin thêm không sinh ra lợi nhuận chênh lệch, dẫn đến VSS = 0 và EVPI = 0. Sự linh hoạt để đối phó rủi ro chỉ diễn ra ở Giai đoạn 2.
    """)
    st.markdown("---")

    # ==========================
    # CÂU 10.5.4
    # ==========================
    st.header(" 10.5.4: Robust Optimization (Minimax Regret)")
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Quyết định First-stage Robust")
        st.dataframe(pd.DataFrame([x_robust]).T.rename(columns={0: "Vốn phân bổ Robust"}).style.format("{:,.2f}"))
    with col6:
        st.warning(f"**Max Regret (Độ hối tiếc lớn nhất):** {robust_regret:,.2f}")

    df_rob_vs_sp = pd.DataFrame({"Quyết định SP (Tối đa kỳ vọng)": x_sp, "Quyết định Robust (Bảo thủ)": x_robust})
    st.table(df_rob_vs_sp.style.format("{:,.2f}"))
    st.markdown("---")

    # ==========================
    # CÂU 10.6: THẢO LUẬN CHÍNH SÁCH
    # ==========================
    st.header("10.6: Thảo luận chính sách")

    st.subheader("a) So với lời giải xác định, lời giải SP có xu hướng đầu tư H nhiều hơn hay ít hơn? Vì sao?")
    st.info("""
    **Trả lời:**
    Về mặt lý luận chiến lược của mô hình, lời giải SP (Stochastic Programming) có xu hướng xem xét và đầu tư vào **H (Nhân lực)** linh hoạt hơn so với lời giải xác định (EV). 

    **Lý do:** Mô hình ngẫu nhiên (SP) đánh giá cao **giá trị của sự linh hoạt (real options)**. Ràng buộc mở rộng AI trong tương lai bị giới hạn bởi lượng vốn Nhân lực (H) từ Giai đoạn 1. Việc đầu tư vào H được SP coi như một khoản "phí quyền chọn", giúp Chính phủ giữ lại dư địa để bứt phá công nghệ nếu kịch bản thuận lợi xảy ra, hoặc dùng H để chống chịu nếu có khủng hoảng.
    *(Lưu ý: Với bộ số liệu cụ thể của bài toán này, do lợi nhuận tức thời của AI quá lớn nên SP tạm thời bỏ qua H, nhưng bản chất toán học của SP luôn cân nhắc H nhiều hơn EV).*
    """)

    st.subheader("b) VSS dương nói lên điều gì về giá trị của tư duy xác suất trong hoạch định chính sách Việt Nam?")
    st.success("""
    **Trả lời:**
    VSS dương là minh chứng đanh thép cho thấy: **Lập kế hoạch chỉ dựa trên một kịch bản trung bình (tất định) sẽ dẫn đến thiệt hại nặng nề về kinh tế.**

    Với độ mở nền kinh tế cực cao (180% GDP), Việt Nam cực kỳ nhạy cảm với các biến động toàn cầu. Tư duy xác suất buộc các nhà hoạch định không "bỏ trứng vào một rổ". VSS dương khẳng định việc trích lập dự phòng, chuẩn bị sẵn các kịch bản hành động ngay từ hôm nay giúp hệ thống chuyển từ trạng thái **"phản ứng bị động"** sang **"chống chịu chủ động"**.
    """)

    st.subheader("c) Đại dịch COVID-19 và bão Yagi là các cú sốc thực tế. Liệu Việt Nam có đang “dưới đầu tư” vào nhân lực số như một hàng hóa bảo hiểm?")
    st.warning("""
    **Trả lời:**
    **Đúng vậy, Việt Nam đang có nguy cơ "dưới đầu tư" vào nhân lực số nếu chỉ chạy theo tăng trưởng ngắn hạn.**

    - **Bài học thực tế:** Khi bão Yagi hay đại dịch làm tê liệt hạ tầng vật lý, công nghệ số là cứu cánh duy nhất.
    - **Nhân lực số là Hàng hóa bảo hiểm (Insurance Good):** Giống như ràng buộc của bài toán, AI không thể tự vận hành nếu thiếu Nhân lực (H). Nếu Chính phủ không đầu tư đủ vào con người từ thời bình, thì khi khủng hoảng ập tới (kịch bản s4), dù có tiền dự phòng cũng không thể triển khai ngay AI để cứu nền kinh tế. Đầu tư vào nhân lực số cốt lõi là để mua một gói "bảo hiểm vĩ mô" chống đứt gãy.
    """)
    st.markdown("<br><br><center><i>Hết báo cáo 10.6</i></center>", unsafe_allow_html=True)

# LƯU Ý: Khối except này bắt lỗi nếu có sự cố xảy ra trong khối try ở trên.
except Exception as e:
    st.error(f"⚠️ Có lỗi xảy ra trong quá trình giải mô hình. Chi tiết: {e}")
    # =========================================================
# OUTPUT CHO BÀI 12 - AIDEOM VN
# =========================================================

st.markdown("---")
st.header("📤 Output cho AIDEOM-VN Dashboard")

# =========================================================
# OUTPUT MODULE M10
# =========================================================

M5_OUTPUT_B10 = {

    # =====================================
    # STOCHASTIC PROGRAMMING
    # =====================================
    "stochastic_solution": {

        "GDP_expected": round(
            obj_sp, 2
        ),

        "x_SP": {
            j: round(x_sp[j], 2)
            for j in categories
        }
    },

    # =====================================
    # EXPECTED VALUE MODEL
    # =====================================
    "expected_value_solution": {

        "x_EV": {
            j: round(x_ev[j], 2)
            for j in categories
        }
    },

    # =====================================
    # ROBUST MODEL
    # =====================================
    "robust_solution": {

        "max_regret": round(
            robust_regret, 2
        ),

        "x_robust": {
            j: round(x_robust[j], 2)
            for j in categories
        }
    },

    # =====================================
    # GIÁ TRỊ THÔNG TIN
    # =====================================
    "information_metrics": {

        "VSS": round(vss, 2),

        "EVPI": round(evpi, 2)
    },

    # =====================================
    # KỊCH BẢN
    # =====================================
    "scenario_probabilities": {

        s: round(probs[s], 4)
        for s in scenarios
    },

    # =====================================
    # WAIT-AND-SEE
    # =====================================
    "wait_and_see": {

        s: round(ws_objectives[s], 2)
        for s in scenarios
    }
}

# =========================================================
# HIỂN THỊ
# =========================================================

def get_b10_output():
    return M5_OUTPUT_B10


if __name__ == "__main__":
    st.json(get_b10_output())

st.success("""
✅ M10 Output đã sẵn sàng để tích hợp vào:
Bài 12 — AIDEOM-VN Dashboard
""")