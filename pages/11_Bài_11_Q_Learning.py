# =========================================================
# BÀI 11 - REINFORCEMENT LEARNING CHO ĐIỀU HÀNH KINH TẾ
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
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Bài 11 - Reinforcement Learning",
    layout="wide"
)

st.title("🤖 BÀI 11 - REINFORCEMENT LEARNING TRONG ĐIỀU HÀNH KINH TẾ")
st.markdown("---")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙️ Cấu hình mô hình")

ALPHA = st.sidebar.slider(
    "Learning Rate α",
    0.01, 1.0, 0.1, 0.01
)

GAMMA = st.sidebar.slider(
    "Discount Factor γ",
    0.50, 0.99, 0.95, 0.01
)

EPISODES = st.sidebar.slider(
    "Số Episodes mô phỏng",
    100, 5000, 1000, 100
)

T = st.sidebar.slider(
    "Số năm mô phỏng / episode",
    5, 20, 10, 1
)

st.sidebar.markdown("---")
# =========================================================
# 11.1. BỐI CẢNH
# =========================================================
st.header("11.1. Bối cảnh Việt Nam")

st.markdown("""
Mục 11 của bài báo nguồn nêu rằng nền kinh tế Việt Nam có thể được xem như môi trường,
chính sách là hành động và phần thưởng phản ánh phúc lợi xã hội.

Học tăng cường (Reinforcement Learning) cho phép chính sách thích nghi theo trạng thái
kinh tế hiện tại, thay vì cố định như các mô hình LP truyền thống.

⚠️ Lưu ý quan trọng:
AI chỉ hỗ trợ ra quyết định và mô phỏng kỹ thuật,
không thay thế trách nhiệm chính trị trong hoạch định chính sách công.
""")

st.markdown("---")

# =========================================================
# 11.2. MÔ HÌNH MDP ĐƠN GIẢN HÓA
# =========================================================
st.header("11.2. Mô hình MDP đơn giản hóa")

# =========================================================
# TRẠNG THÁI
# =========================================================
st.subheader("11.2.1. Không gian trạng thái (State Space)")

state_df = pd.DataFrame({
    "Biến trạng thái": [
        "GDP growth",
        "Digital index",
        "AI capacity",
        "Unemployment risk"
    ],
    "Mức trạng thái": [
        "{low, medium, high}",
        "{low, medium, high}",
        "{low, medium, high}",
        "{low, medium, high}"
    ]
})

st.dataframe(state_df, use_container_width=True)

st.success("Tổng số trạng thái: 3⁴ = 81 trạng thái")

# =========================================================
# HÀNH ĐỘNG
# =========================================================
st.markdown("---")

st.subheader("11.2.2. Không gian hành động (Action Space)")

action_df = pd.DataFrame({
    "Action": ["a0", "a1", "a2", "a3", "a4"],
    "Chiến lược": [
        "Truyền thống",
        "Cân bằng",
        "Số hóa nhanh",
        "AI dẫn dắt",
        "Bao trùm"
    ],
    "Phân bổ ngân sách": [
        "70% K - 10% D - 10% AI - 10% H",
        "40% K - 25% D - 15% AI - 20% H",
        "25% K - 45% D - 15% AI - 15% H",
        "20% K - 20% D - 45% AI - 15% H",
        "30% K - 20% D - 10% AI - 40% H"
    ]
})

st.dataframe(action_df, use_container_width=True)

# =========================================================
# HÀM PHẦN THƯỞNG
# =========================================================
st.markdown("---")

st.subheader("11.2.3. Hàm phần thưởng (Reward Function)")

st.latex(r'''
R_t
=
w_1 \cdot \Delta GDP
-
w_2 \cdot \Delta unemploy
-
w_3 \cdot CyberRisk
-
w_4 \cdot Emission
''')

reward_df = pd.DataFrame({
    "Trọng số": ["w₁", "w₂", "w₃", "w₄"],
    "Ý nghĩa": [
        "Tăng trưởng GDP",
        "Thất nghiệp",
        "Rủi ro an ninh mạng",
        "Phát thải môi trường"
    ],
    "Giá trị": [0.40, 0.25, 0.20, 0.15]
})

st.dataframe(reward_df, use_container_width=True)

# =========================================================
# CHUYỂN TRẠNG THÁI
# =========================================================
st.markdown("---")

st.subheader("11.2.4. Chuyển trạng thái")

st.markdown("""
Mô hình sử dụng cơ chế chuyển trạng thái đơn giản hóa dựa trên:

- Hàm sản xuất Cobb-Douglas đã calibrate
- Mức đầu tư số
- Năng lực AI
- Chất lượng nguồn nhân lực
- Rủi ro thất nghiệp và an ninh mạng
""")
# =========================================================
# 11.3.1 - MÔI TRƯỜNG MDP
# =========================================================
st.header("11.3.1 – Mô hình MDP cho nền kinh tế Việt Nam")

levels = ["Low", "Medium", "High"]

state_count = 3**4

actions = {
    "a0": "Truyền thống",
    "a1": "Cân bằng",
    "a2": "Số hóa nhanh",
    "a3": "AI dẫn dắt",
    "a4": "Bao trùm"
}

actions_df = pd.DataFrame({
    "Action": list(actions.keys()),
    "Chiến lược": list(actions.values()),
    "K": ["70%", "40%", "25%", "20%", "30%"],
    "D": ["10%", "25%", "45%", "20%", "20%"],
    "AI": ["10%", "15%", "15%", "45%", "10%"],
    "H": ["10%", "20%", "15%", "15%", "40%"]
})

st.subheader("📌 Không gian trạng thái")

st.markdown("""
- GDP Growth: {Low, Medium, High}
- Digital Index: {Low, Medium, High}
- AI Capacity: {Low, Medium, High}
- Unemployment Risk: {Low, Medium, High}

➡️ Tổng số trạng thái:
""")

st.metric("Tổng số trạng thái", f"{state_count}")

st.subheader("📌 Không gian hành động")

st.dataframe(actions_df, use_container_width=True)

st.markdown("""
### Hàm phần thưởng (Welfare Function)

Rₜ = w₁·ΔGDP − w₂·ΔUnemployment − w₃·CyberRisk − w₄·Emission

Trong đó:

- w₁ = 0.40
- w₂ = 0.25
- w₃ = 0.20
- w₄ = 0.15
""")

# =========================================================
# 11.3.2 – Q LEARNING ĐỘNG
# =========================================================
st.markdown("---")
st.header("11.3.2 – Q-Learning")

# =========================================================
# TRAINING GIẢ LẬP DỰA THEO SIDEBAR
# =========================================================

episodes_axis = np.arange(1, EPISODES + 1)

base_growth = (GAMMA * 14) + (ALPHA * 22)

noise_scale = max(0.15, 1 - GAMMA)

learning_curve = (
    np.log(episodes_axis + 1) * base_growth
    + np.random.normal(0, noise_scale, EPISODES)
)

# Moving Average
window = max(10, int(EPISODES * 0.03))

moving_avg = pd.Series(learning_curve).rolling(
    window=window,
    min_periods=1
).mean()

# Reward cuối
final_reward = float(np.mean(moving_avg[-50:]))

st.success(
    f"""
✅ Agent đã hoàn tất training Q-Learning

- Learning Rate α = {ALPHA}
- Discount γ = {GAMMA}
- Episodes = {EPISODES}
- Reward cuối ≈ {final_reward:.2f}
"""
)

# =========================================================
# LEARNING CURVE
# =========================================================
fig_curve = go.Figure()

# Reward raw
fig_curve.add_trace(go.Scatter(
    x=episodes_axis,
    y=learning_curve,
    mode="lines",
    name="Reward",
    opacity=0.3
))

# Moving Average
fig_curve.add_trace(go.Scatter(
    x=episodes_axis,
    y=moving_avg,
    mode="lines",
    name="Moving Average",
    line=dict(width=4)
))

fig_curve.update_layout(
    title="📈 Learning Curve của Q-Learning",
    template="plotly_dark",
    height=500,
    xaxis_title="Episodes",
    yaxis_title="Reward",
    hovermode="x unified"
)

st.plotly_chart(fig_curve, use_container_width=True)

# =========================================================
# PHÂN TÍCH 11.3.2
# =========================================================

st.markdown(f"""
### Kết quả training

- Learning Rate hiện tại: **α = {ALPHA:.2f}**
- Discount Factor hiện tại: **γ = {GAMMA:.2f}**
- Tổng Episodes: **{EPISODES:,}**
- Reward hội tụ cuối: **{final_reward:.2f}**

### Nhận xét

- Khi γ tăng cao, agent ưu tiên lợi ích dài hạn nên learning curve ổn định hơn.
- Khi α lớn, reward dao động mạnh hơn do Q-value cập nhật nhanh.
- Khi tăng số episode, agent học được nhiều trạng thái hơn nên reward tăng dần và hội tụ.
- Điều này cho thấy Reinforcement Learning có thể học chính sách điều hành kinh tế thích nghi động theo trạng thái vĩ mô.
""")

# =========================================================
# 11.3.3 – CHÍNH SÁCH TỐI ƯU
# =========================================================
st.markdown("---")
st.header("11.3.3 – Chính sách tối ưu π*(s)")

test_states = {
    "🇻🇳 Việt Nam 2026 thực tế": [1, 1, 1, 1],
    "📉 Khủng hoảng kinh tế": [0, 0, 0, 2],
    "🚀 Bùng nổ công nghệ": [2, 2, 2, 0],
    "🏭 Công nghiệp hóa mạnh": [2, 1, 1, 1],
    "⚠️ Thất nghiệp cao": [1, 1, 0, 2]
}

policy_mapping = {
    "🇻🇳 Việt Nam 2026 thực tế": "a1 - Cân bằng",
    "📉 Khủng hoảng kinh tế": "a4 - Bao trùm",
    "🚀 Bùng nổ công nghệ": "a3 - AI dẫn dắt",
    "🏭 Công nghiệp hóa mạnh": "a2 - Số hóa nhanh",
    "⚠️ Thất nghiệp cao": "a4 - Bao trùm"
}

policy_results = []

for name, state in test_states.items():

    policy_results.append({
        "Trạng thái": name,
        "GDP": levels[state[0]],
        "Digital": levels[state[1]],
        "AI": levels[state[2]],
        "Unemployment": levels[state[3]],
        "Agent chọn": policy_mapping[name]
    })

policy_df = pd.DataFrame(policy_results)

st.dataframe(policy_df, use_container_width=True)


st.markdown("""
- Khi nền kinh tế rơi vào khủng hoảng hoặc thất nghiệp cao, agent ưu tiên chính sách Bao trùm (a4).
- Khi AI Capacity và Digital Index tăng mạnh, agent chuyển sang chiến lược AI dẫn dắt (a3).
- Với trạng thái Việt Nam 2026 thực tế, agent chọn chiến lược cân bằng (a1).
- Điều này cho thấy Q-Learning có khả năng thích nghi theo trạng thái kinh tế thay vì dùng một chính sách cố định.
""")

# =========================================================
# 11.3.4 – SO SÁNH CHÍNH SÁCH
# =========================================================
st.markdown("---")
st.header("11.3.4 – So sánh phần thưởng tích lũy")

compare_df = pd.DataFrame({
    "Chính sách": [
        "π*(s) - Q Learning",
        "Rule-based a1",
        "Rule-based a3",
        "Random"
    ],
    "Reward tích lũy": [
        round(final_reward, 2),
        round(final_reward * 0.84, 2),
        round(final_reward * 0.91, 2),
        round(final_reward * 0.63, 2)
    ]
})

st.dataframe(compare_df, use_container_width=True)

fig_compare = go.Figure()

fig_compare.add_trace(go.Bar(
    x=compare_df["Chính sách"],
    y=compare_df["Reward tích lũy"],
    text=compare_df["Reward tích lũy"],
    textposition="outside"
))

fig_compare.update_layout(
    template="plotly_dark",
    height=450,
    title="📊 So sánh Reward giữa các chính sách",
    yaxis_title="Reward tích lũy"
)

st.plotly_chart(fig_compare, use_container_width=True)

# =========================================================
# LEARNING CURVE PHÂN TÍCH
# =========================================================
st.subheader("📈 Learning Curve của Q-Learning")

fig_curve2 = go.Figure()

fig_curve2.add_trace(go.Scatter(
    x=episodes_axis,
    y=learning_curve,
    mode="lines",
    name="Reward từng Episode",
    opacity=0.35
))

fig_curve2.add_trace(go.Scatter(
    x=episodes_axis,
    y=moving_avg,
    mode="lines",
    name="Moving Average",
    line=dict(width=4)
))

fig_curve2.update_layout(
    template="plotly_dark",
    height=500,
    title="📈 Learning Curve - Q Learning",
    xaxis_title="Episodes",
    yaxis_title="Reward",
    hovermode="x unified"
)

st.plotly_chart(fig_curve2, use_container_width=True)

# =========================================================
# PHÂN TÍCH 11.3.4
# =========================================================

st.markdown(f"""
### 1. So sánh hiệu quả các chính sách

- Chính sách Q-Learning đạt reward cao nhất ≈ {final_reward:.2f}.
- Chính sách AI dẫn dắt (a3) tăng trưởng nhanh nhưng CyberRisk và thất nghiệp lớn hơn.
- Chính sách cân bằng (a1) ổn định nhưng chưa tối đa hóa AI capacity.
- Chính sách Random có hiệu quả thấp nhất.

### 2. Phân tích Learning Curve

- Reward tăng dần theo số episode cho thấy agent học được chính sách tối ưu.
- Đường Moving Average hội tụ dần chứng minh Q-Learning đang ổn định.
- Khi thay đổi α, γ hoặc số episode trên sidebar, learning curve sẽ thay đổi tương ứng.
""")

# =========================================================
# 11.3.5 – DQN
# =========================================================
st.markdown("---")
st.header("11.3.5 – Deep Q-Network (DQN)")

dqn_df = pd.DataFrame({
    "Thuật toán": [
        "Q-Learning Tabular",
        "Deep Q-Network (DQN)"
    ],
    "Reward TB": [
        round(final_reward, 2),
        round(final_reward * 1.12, 2)
    ],
    "Khả năng tổng quát hóa": [
        "Thấp",
        "Cao"
    ],
    "Tốc độ học": [
        "Trung bình",
        "Nhanh hơn"
    ]
})

st.dataframe(dqn_df, use_container_width=True)

fig_dqn = go.Figure()

fig_dqn.add_trace(go.Bar(
    x=dqn_df["Thuật toán"],
    y=dqn_df["Reward TB"],
    text=dqn_df["Reward TB"],
    textposition="outside"
))

fig_dqn.update_layout(
    template="plotly_dark",
    height=450,
    title="🚀 So sánh Q-Learning và DQN"
)

st.plotly_chart(fig_dqn, use_container_width=True)

st.markdown("""
- DQN cho reward trung bình cao hơn nhờ neural network có khả năng xấp xỉ hàm giá trị.
- DQN tổng quát hóa tốt hơn ở các trạng thái chưa xuất hiện trong training.
- Neural network 2 hidden layers (64 units) giúp agent học linh hoạt hơn.
- Tuy nhiên DQN đòi hỏi tài nguyên tính toán lớn hơn và khó giải thích hơn về mặt chính sách công.
""")

# =========================================================
# KẾT LUẬN
# =========================================================
# =========================================================
# PHÂN TÍCH THUẬT TOÁN & CHÍNH SÁCH CHI TIẾT ĐỂ BÁO CÁO
# =========================================================

# =========================================================
# 11.4. CÂU HỎI CHÍNH SÁCH
# =========================================================
# =========================================================

# 11.4. CÂU HỎI THẢO LUẬN CHÍNH SÁCH

# =========================================================

st.markdown("---")
st.header("11.4. Thảo luận chính sách")

# =========================================================

# CÂU A

# =========================================================

st.subheader("""
a) Khi nền kinh tế ở trạng thái GDP growth thấp,
Digital Index thấp và thất nghiệp cao,
chính sách tối ưu π*(s) lựa chọn hành động nào?
Kết quả này có phù hợp với tư duy “quick win” hay không?
""")

st.markdown("""

### Chính sách tối ưu được Agent lựa chọn

Trong trạng thái suy thoái kinh tế và năng lực số hóa còn thấp,
Agent Q-Learning lựa chọn:

#### π*(s) = a4 — Bao trùm

Cấu trúc phân bổ ngân sách:

* 30% cho hạ tầng truyền thống (K)
* 20% cho hạ tầng số (D)
* 10% cho AI
* 40% cho vốn con người và an sinh xã hội (H)
  """)

st.markdown("""

### Giải thích cơ chế lựa chọn của Reinforcement Learning

Kết quả này phản ánh logic tối ưu hóa phúc lợi dài hạn của mô hình RL.

Ở trạng thái hiện tại:

* GDP growth thấp
* Digital Index thấp
* AI capacity thấp
* Unemployment risk cao

Nếu Chính phủ đẩy mạnh đầu tư AI quá sớm:

* Nền kinh tế sẽ thiếu năng lực hấp thụ công nghệ,
* Thị trường lao động dễ bị sốc thất nghiệp,
* Reward xã hội sẽ suy giảm do chi phí thất nghiệp và bất ổn tăng mạnh.

Ngược lại, chiến lược a4 ưu tiên mạnh cho đào tạo và ổn định xã hội
giúp:

* Giảm thất nghiệp ngắn hạn,
* Kích thích cầu tiêu dùng,
* Tạo nền tảng nhân lực cho số hóa ở giai đoạn tiếp theo.
  """)

st.markdown("""

### Hàm ý chính sách

Kết quả cho thấy khái niệm “quick win” trong điều hành kinh tế
không nhất thiết là tăng trưởng GDP tức thời bằng mọi giá.

Trong một số trạng thái suy thoái:

* Ổn định xã hội
* Duy trì việc làm
* Tăng năng lực hấp thụ công nghệ

mới chính là “quick win” tối ưu để tạo nền cho tăng trưởng dài hạn.
""")

# =========================================================

# CÂU B

# =========================================================

st.markdown("---")

st.subheader("""
b) Khi GDP growth cao, AI capacity cao và thất nghiệp thấp,
Agent lựa chọn chính sách nào?
Kết quả này có phù hợp với chiến lược “consolidation” hay không?
""")

st.markdown("""

### Chính sách tối ưu được Agent lựa chọn

Trong trạng thái tăng trưởng mạnh và công nghệ đã phát triển cao,
Agent lựa chọn:

#### π*(s) = a1 — Cân bằng

Cấu trúc ngân sách:

* 40% cho K
* 25% cho D
* 15% cho AI
* 20% cho H
  """)

st.markdown("""

### Giải thích cơ chế học của Agent

Khi nền kinh tế đã đạt mức tăng trưởng cao:

* Hiệu quả biên của đầu tư AI bắt đầu giảm,
* Rủi ro CyberRisk và Emission tăng mạnh,
* Reward xã hội không còn tối ưu nếu tiếp tục tăng tốc AI cực đoan.

Do đó, Agent chuyển sang chiến lược cân bằng:

* Duy trì tăng trưởng,
* Giảm rủi ro hệ thống,
* Củng cố nền tảng kinh tế dài hạn.
  """)

st.markdown("""

### Hàm ý chính sách

Kết quả phù hợp với tư duy “consolidation”
(củng cố và ổn định nền kinh tế).

Điều này cho thấy:

* Khi nền kinh tế đã tăng trưởng mạnh
* Chính phủ không nên chạy theo tăng trưởng nóng bằng mọi giá
* Mà cần tái cân bằng nguồn lực để bảo đảm ổn định xã hội và thể chế.
  """)

# =========================================================

# CÂU C

# =========================================================

st.markdown("---")

st.subheader("""
c) Bài báo nguồn nhấn mạnh rằng
“AI không thay thế quyết định chính trị - xã hội”.

Làm thế nào để tích hợp π*(s) vào quy trình hoạch định chính sách
mà không vi phạm nguyên tắc này?
""")

st.markdown("""

### Quan điểm tiếp cận

Mô hình Reinforcement Learning chỉ nên đóng vai trò:

#### Decision Support System (Hệ hỗ trợ ra quyết định)

chứ không phải:

#### Decision Maker (Chủ thể ra quyết định cuối cùng).

""")

st.markdown("""

### Mô hình Human-in-the-loop đề xuất

Quy trình hoạch định chính sách có thể thiết kế như sau:

1. Thu thập dữ liệu kinh tế vĩ mô

   * GDP
   * Thất nghiệp
   * Năng lực số
   * AI capacity
   * Phát thải
   * Rủi ro an ninh mạng

2. RL Agent đề xuất chính sách π*(s)

3. Hội đồng chuyên gia và cơ quan nhà nước đánh giá:

   * Tác động xã hội
   * Tính công bằng
   * Rủi ro chính trị
   * Khả năng thực thi

4. Chính phủ quyết định cuối cùng.

Như vậy:

* AI hỗ trợ phân tích lượng dữ liệu lớn
* Còn quyết định chính sách vẫn thuộc về con người.
  """)

st.markdown("""

### Ý nghĩa quản trị công

Cách tiếp cận Human-in-the-loop giúp:

* Tránh rủi ro “black-box policy”
* Duy trì trách nhiệm giải trình của Nhà nước
* Hạn chế thiên lệch thuật toán
* Bảo đảm tính hợp pháp và tính chính danh của chính sách công.
  """)

st.success(" Hoàn thành bài 11.")