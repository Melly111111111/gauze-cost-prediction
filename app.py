import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LassoCV
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

# 页面配置：使用宽屏模式
st.set_page_config(page_title="纱布产品成本智能决策系统", layout="wide", initial_sidebar_state="expanded")

# 修复后的 CSS 样式块
st.markdown("""
<style>
    .main { background-color: #f5f7f9; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #007bff; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def train_model():
    file_path = '纱布AI报价数据1.xlsx'
    try:
        df = pd.read_excel(file_path)
        target_col = '成本单价PCS'
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LassoCV(alphas=np.logspace(-6, 2, 200), cv=LeaveOneOut(), max_iter=50000)
        model.fit(X_scaled, y)
        return model, scaler, X.columns.tolist(), df
    except:
        return None, None, None, None


model, scaler, all_feature_names, raw_df = train_model()

# --- 侧边栏：配置与说明 ---
with st.sidebar:
    st.image("https://www.streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=150)
    st.title("系统设置")
    st.info("本系统对生产工艺参数进行预测。它可以自动识别非线性关系并剔除无关干扰因素。")
    if raw_df is not None:
        st.write(f"训练样本数：{len(raw_df)}")
        st.write(f"特征维度：{len(all_feature_names)}")

st.title("纱布产品成本智能决策系统")
st.markdown("---")

if model:
    # 第一部分：参数输入区
    with st.expander("🛠第一步：录入工艺参数", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            v_bg = st.number_input("包高（cm）", value=8.50, format="%.2f")
            v_zj = st.number_input("粘胶配比%", value=0.50, format="%.2f", step=0.01)
            v_mx = st.number_input("每箱数量", value=5000, step=100)
        with c2:
            v_jx = st.number_input("机型（cm）", value=4.80, format="%.2f")
            v_dl = st.number_input("涤纶配比%", value=0.50, format="%.2f", step=0.01)
            v_cn = st.number_input("小时产能", value=86331, step=1000)
        with c3:
            v_kz = st.number_input("克重g/㎡", value=30.00, format="%.2f")
            v_kl = st.number_input("开料cm", value=9.20, format="%.2f")
            v_rs = st.number_input("岗位人数", value=4.60, format="%.2f", step=0.01)

        predict_btn = st.button("开始智能核价", use_container_width=True, type="primary")

    if predict_btn:
        # 准备数据
        user_input = {"包高（cm）": v_bg, "机型（cm）": v_jx, "克重g/㎡": v_kz, "粘胶配比%": v_zj,
                      "涤纶配比%": v_dl, "开料cm": v_kl, "每箱数量": v_mx, "小时产能": v_cn, "岗位人数": v_rs}
        full_input_df = pd.DataFrame(0.0, index=[0], columns=all_feature_names)
        for k, v in user_input.items():
            if k in all_feature_names: full_input_df[k] = v

        # 预测
        input_scaled = scaler.transform(full_input_df)
        prediction = model.predict(input_scaled)[0]
        final_price = max(0, prediction)

        # 第二部分：结果展示区
        st.markdown("### 核价结果分析")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            # 【修改点】将 stdio 改为 html
            st.markdown('<div class="predict-box">', unsafe_allow_html=True)

            st.metric("预估成本单价 (PCS)", f"¥ {final_price:.6f}")

            # 计算较均值的偏差
            avg_price = raw_df['成本单价PCS'].mean()
            delta = ((final_price - avg_price) / avg_price) * 100
            st.write(f"相对于历史平均单价: **{delta:+.2f}%**")

            # 【修改点】将 stdio 改为 html
            st.markdown('</div>', unsafe_allow_html=True)

            # 添加建议
            if delta > 10:
                st.warning("⚠️ 当前配置成本偏高，建议优化产能或岗位配比。")
            else:
                st.success("✅ 当前成本控制在合理范围内。")

        with res_col2:
            # 绘制参数贡献度仪表盘
            # 获取 Lasso 系数，展示当前影响最大的前 5 个因素
            coef_df = pd.DataFrame({'特征': all_feature_names, '影响权重': model.coef_})
            coef_df = coef_df[coef_df['影响权重'] != 0].sort_values(by='影响权重', ascending=False)

            fig_coef = px.bar(coef_df, x='影响权重', y='特征', orientation='h',
                              title="哪些因素在影响价格？(模型系数分析)",
                              color='影响权重', color_continuous_scale='RdYlGn_r')
            fig_coef.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_coef, use_container_width=True)

        # 第三部分：数据分布对比
        st.markdown("---")
        st.markdown("### 行业对标：当前输入在历史数据中的位置")

        # 绘制价格分布图并标记当前预测值
        fig_dist = px.histogram(raw_df, x="成本单价PCS", nbins=20, title="历史报价分布图",
                                labels={'成本单价PCS': '单价'}, color_discrete_sequence=['#AEC6CF'])
        fig_dist.add_vline(x=final_price, line_width=3, line_dash="dash", line_color="red",
                           annotation_text="当前预测值", annotation_position="top left")

        st.plotly_chart(fig_dist, use_container_width=True)

        st.balloons()  # 撒花庆祝
else:
    st.error("无法加载模型，请检查 Excel 数据文件。")
