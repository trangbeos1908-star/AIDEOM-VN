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
import plotly.express as px
st.set_page_config(
    page_title="AIDEOM-VN Dashboard",
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
st.title("🇻🇳 Bài 12 — AIDEOM-VN Dashboard tích hợp")
st.markdown("""
Mô hình AIDEOM-VN tích hợp:

- M1: Dự báo kinh tế
- M2: Đánh giá sẵn sàng số
- M3: Tối ưu phân bổ
- M4: Mô phỏng lao động AI
- M5: Đánh giá rủi ro
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan",
    "💰 Phân bổ",
    "📈 Kịch bản",
    "⚠️ Rủi ro"
])

with tab1:
    st.header("📊 Tổng quan")

    m1 = {
        "GDP_2030": 18261.968550744412,
        "Growth_2030": 7.286509683599407,
        "TFP_2030": 37.05932093160151,
        "Digital_Contribution": 10.370091821739631,
        "AI_Contribution": 6.2384896145361886,
        "Human_Contribution": 2.870040817536204
    }

    m2 = {
    "best_region": "Đông Nam Bộ",
    "digital_index": 94.02,
    "top3_regions": [
        "Đông Nam Bộ",
        "Đồng bằng sông Hồng",
        "Bắc Trung Bộ & DH miền Trung"
    ],
    "ai_readiness_score": 41.66
}

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "GDP 2030",
        f"{m1['GDP_2030']:,.0f}"
    )

    c2.metric(
        "Growth 2030 (%)",
        round(m1["Growth_2030"], 2)
    )

    c3.metric(
        "Digital Index",
        m2["digital_index"]
    )

    c4.metric(
        "AI Readiness",
        m2["ai_readiness_score"]
    )

    factor_df = pd.DataFrame({
        "Factor": [
            "Digital",
            "AI",
            "Human"
        ],
        "Contribution": [
            m1["Digital_Contribution"],
            m1["AI_Contribution"],
            m1["Human_Contribution"]
        ]
    })

    fig = px.bar(
        factor_df,
        x="Factor",
        y="Contribution",
        title="Đóng góp các yếu tố vào tăng trưởng"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("🌐 M2 — Đánh giá sẵn sàng số")

    st.success(
        f"Vùng dẫn đầu: {m2['best_region']}"
    )

    top_region_df = pd.DataFrame({
        "Xếp hạng": [1, 2, 3],
        "Vùng": m2["top3_regions"]
    })

    st.dataframe(
        top_region_df,
        use_container_width=True
    )
    st.markdown("---")
st.subheader("📋 Tóm tắt thiết kế hệ thống")

system_df = pd.DataFrame({
    "Module": ["M1", "M2", "M3", "M4", "M5", "M6"],
    "Tên": [
        "Dự báo kinh tế",
        "Sẵn sàng số",
        "Tối ưu phân bổ",
        "Lao động",
        "Rủi ro",
        "Dashboard"
    ],
    "Đầu vào": [
        "Macro 2020-2025",
        "Sectors, Regions",
        "Budget, β-matrix",
        "x_AI, x_H",
        "Risk params",
        "Outputs M1-M5"
    ],
    "Đầu ra": [
        "GDP, TFP 2030",
        "Digital + AI Index",
        "Phân bổ ngành-vùng",
        "NetJob từng ngành",
        "Cyber, Env, Dependency",
        "Trực quan kịch bản"
    ],
    "Kỹ thuật": [
        "Cobb-Douglas (Bài 1)",
        "TOPSIS (Bài 6)",
        "LP (Bài 4) + Dynamic (Bài 8)",
        "LP (Bài 9)",
        "NSGA-II (Bài 7) + SP (Bài 10)",
        "Streamlit + Plotly"
    ]
})

st.dataframe(
    system_df,
    use_container_width=True,
    hide_index=True
)
with tab2:
    st.header("💰 M3 — Tối ưu phân bổ ngân sách")

    m3_b4 = {
        "GDP_gain_optimal": 52485,
        "total_budget": 50000,
        "budget_allocation": {
            "NMM": {"I": 0, "D": 9700, "AI": 0, "H": 2300},
            "RRD": {"I": 0, "D": 0, "AI": 5000, "H": 0},
            "NCC": {"I": 0, "D": 1200, "AI": 0, "H": 3800},
            "CH": {"I": 0, "D": 12000, "AI": 0, "H": 0},
            "SE": {"I": 0, "D": 0, "AI": 5400, "H": 0},
            "MD": {"I": 0, "D": 4700, "AI": 0, "H": 5900}
        }
    }

    m3_b8 = {
        "welfare": 52.5922,
        "status": "optimal",
        "I_K": {"2026": 0, "2027": 0, "2028": 0, "2029": 0, "2030": 0, "2031": 0, "2032": 0, "2033": 0, "2034": 0, "2035": 0},
        "I_D": {"2026": 82.94, "2027": 54.97, "2028": 52.15, "2029": 53.64, "2030": 29.64, "2031": 0.02, "2032": 0, "2033": 0, "2034": 0, "2035": 0},
        "I_AI": {"2026": 2.61, "2027": 41.5, "2028": 39.57, "2029": 40.47, "2030": 29.82, "2031": 15.95, "2032": 0, "2033": 0, "2034": 0, "2035": 0},
        "I_H": {"2026": 53.4, "2027": 52.57, "2028": 34.64, "2029": 0, "2030": 0, "2031": 0, "2032": 0, "2033": 0, "2034": 0, "2035": 0}
    }

    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng ngân sách", f"{m3_b4['total_budget']:,.0f}")
    c2.metric("GDP gain tối ưu", f"{m3_b4['GDP_gain_optimal']:,.0f}")
    c3.metric("Welfare động", m3_b8["welfare"])

    allocation_rows = []

    for region, values in m3_b4["budget_allocation"].items():
        for item, amount in values.items():
            allocation_rows.append({
                "Vùng": region,
                "Hạng mục": item,
                "Ngân sách": amount
            })

    allocation_df = pd.DataFrame(allocation_rows)

    st.subheader("Phân bổ ngân sách theo vùng và hạng mục — Bài 4 LP")
    st.dataframe(allocation_df, use_container_width=True)

    fig_bar = px.bar(
        allocation_df,
        x="Vùng",
        y="Ngân sách",
        color="Hạng mục",
        title="Phân bổ ngân sách theo vùng"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    total_by_item = (
        allocation_df
        .groupby("Hạng mục", as_index=False)["Ngân sách"]
        .sum()
    )

    fig_pie = px.pie(
        total_by_item,
        names="Hạng mục",
        values="Ngân sách",
        title="Cơ cấu phân bổ tổng theo I / D / AI / H"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Phân bổ động theo thời gian — Bài 8 Dynamic Optimization")

    dynamic_rows = []

    for year in m3_b8["I_K"].keys():
        dynamic_rows.append({
            "Năm": year,
            "I_K": m3_b8["I_K"][year],
            "I_D": m3_b8["I_D"][year],
            "I_AI": m3_b8["I_AI"][year],
            "I_H": m3_b8["I_H"][year]
        })

    dynamic_df = pd.DataFrame(dynamic_rows)

    st.dataframe(dynamic_df, use_container_width=True)

    dynamic_long = dynamic_df.melt(
        id_vars="Năm",
        value_vars=["I_K", "I_D", "I_AI", "I_H"],
        var_name="Loại đầu tư",
        value_name="Giá trị"
    )

    fig_dynamic = px.line(
        dynamic_long,
        x="Năm",
        y="Giá trị",
        color="Loại đầu tư",
        markers=True,
        title="Quỹ đạo phân bổ động 2026–2035"
    )

    st.plotly_chart(fig_dynamic, use_container_width=True)

with tab3:
    st.header("📈 So sánh 5 kịch bản chính sách")

    scenario_df = pd.DataFrame({
        "Kịch bản": [
            "S1 Truyền thống",
            "S2 Số hóa nhanh",
            "S3 AI dẫn dắt",
            "S4 Bao trùm số",
            "S5 Tối ưu cân bằng"
        ],
        "K": [70, 25, 20, 30, 35],
        "D": [10, 45, 20, 20, 30],
        "AI": [10, 15, 45, 10, 20],
        "H": [10, 15, 15, 40, 15],
        "GDP 2030": [
            17050,
            17880,
            18220,
            17640,
            18261.97
        ],
        "Digital Index": [
            65.2,
            88.7,
            82.4,
            76.5,
            94.02
        ],
        "NetJob": [
            980000,
            1250000,
            1050000,
            1650000,
            1650000
        ],
        "Risk Score": [
            -3200,
            -4100,
            -5554.49,
            -3600,
            -5200
        ]
    })

    st.dataframe(
        scenario_df,
        use_container_width=True
    )

    fig_gdp = px.bar(
        scenario_df,
        x="Kịch bản",
        y="GDP 2030",
        title="So sánh GDP 2030 giữa 5 kịch bản"
    )
    st.plotly_chart(fig_gdp, use_container_width=True)

    fig_job = px.bar(
        scenario_df,
        x="Kịch bản",
        y="NetJob",
        title="So sánh tác động việc làm NetJob"
    )
    st.plotly_chart(fig_job, use_container_width=True)

    fig_risk = px.line(
        scenario_df,
        x="Kịch bản",
        y="Risk Score",
        markers=True,
        title="So sánh Risk Score giữa các kịch bản"
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    st.info(
        "S5 được xem là kịch bản tối ưu cân bằng vì đạt GDP 2030 cao, "
        "Digital Index tốt, NetJob lớn và rủi ro ở mức có thể kiểm soát."
    )
    
with tab4:
    st.header("⚠️ M5 — Đánh giá rủi ro đa mục tiêu")

    m5 = {
        "GDP_gain": 53516.52,
        "Bao_trum": 333.06,
        "Phat_thai": 908.81,
        "Rui_ro": -9953.51,
        "C_star": 0.8799,
        "cyber_risk": -9953.51,
        "environmental_risk": 908.81,
        "dependency_risk": -7962.81,
        "risk_score": -5554.49,
        "num_pareto_solutions": 99,
        "best_C_star": 0.8799
    }

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("GDP gain", f"{m5['GDP_gain']:,.0f}")
    c2.metric("Bao trùm", round(m5["Bao_trum"], 2))
    c3.metric("Phát thải", round(m5["Phat_thai"], 2))
    c4.metric("C*", m5["C_star"])

    risk_df = pd.DataFrame({
        "Loại rủi ro": [
            "Cyber",
            "Environmental",
            "Dependency"
        ],
        "Điểm": [
            m5["cyber_risk"],
            m5["environmental_risk"],
            m5["dependency_risk"]
        ]
    })

    fig = px.bar(
        risk_df,
        x="Loại rủi ro",
        y="Điểm",
        title="Thành phần rủi ro M5"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tổng hợp Pareto")

    pareto_df = pd.DataFrame({
        "Chỉ tiêu": [
            "Số nghiệm Pareto",
            "Best C*",
            "Risk Score"
        ],
        "Giá trị": [
            m5["num_pareto_solutions"],
            m5["best_C_star"],
            m5["risk_score"]
        ]
    })

    st.dataframe(pareto_df, use_container_width=True)

    if m5["environmental_risk"] > 800:
        st.warning("Cảnh báo: rủi ro môi trường/phát thải đang cao.")
    else:
        st.success("Rủi ro môi trường ở mức kiểm soát được.")
    