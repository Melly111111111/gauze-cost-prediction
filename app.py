import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LassoCV
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

# 1. 页面配置
st.set_page_config(page_title="纱布产品成本智能决策系统", layout="wide")

# 2. 深度定制配色 CSS (完全对标图片颜色)
st.markdown("""
<style>
    /* 全局背景色：浅灰蓝 */
    .stApp { background-color: #f8f9fa; }

    /* 顶部标题装饰 */
    h1 { color: #004085; 
        border-bottom: 2px solid #004085; 
        padding-bottom: 10px; 
        text-align: center; /* 让文字居中 */
    }

    /* 卡片式容器 */
    .stExpander, div[data-testid="stVerticalBlock"] > div.chart-card {
        background-color: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* 结果分析框样式 */
    .result-card {
        background-color: #ffffff;
        border: 2px solid #004085;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 25px;
    }

    /* 按钮样式：橙色醒目 */
    div.stButton > button[kind="primary"] {
        background-color: #fd7e14 !important;
        border-color: #fd7e14 !important;
        color: white !important;
        width: 100%;
        height: 50px;
        font-weight: bold;
    }

    /* 重置按钮样式 */
    .btn-reset button {
        background-color: #007bff !important;
        color: white !important;
        border: none !important;
        white-space: nowrap;
    }

    /* 蓝色小标题栏 */
    .section-header {
        background-color: #e9f1f7;
        color: #004085;
        font-weight: bold;
        padding: 5px 10px;
        border-left: 5px solid #004085;
        margin-bottom: 15px;
        text-align: center;
    }

    /* 价格大数字样式 */
    .price-box {
        background-color: #ffffff;
        border: 2px solid #004085;
        border-radius: 10px 10px 0 0;
        padding: 20px;
        text-align: center;
        margin-bottom: 0px;
    }

    /* 偏差显示框 */
    .delta-box {
        background-color: #e9f1f7;
        border-radius: 0 0 10px 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #dee2e6;
        border-top: none;
    }

    /* 智能判断框 */
    .judgment-box {
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
        text-align: center;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .judgment-success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .judgment-warning { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
</style>
""", unsafe_allow_html=True)


# 3. 数据加载与模型训练
@st.cache_resource
def train_model():
    try:
        df = pd.read_excel('纱布AI报价数据1.xlsx')
        target_col = '成本单价PCS'
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LassoCV(alphas=np.logspace(-6, 2, 200), cv=LeaveOneOut(), max_iter=50000)
        model.fit(X_scaled, y)

        return model, scaler, X.columns.tolist(), df
    except Exception as e:
        st.error(f"数据加载或模型训练失败: {e}")
        return None, None, None, None


model, scaler, all_feature_names, raw_df = train_model()

# --- 界面主体 ---
st.title("纱布产品成本智能决策系统")
st.info("本系统对生产工艺参数进行预测,可以自动识别非线性关系并剔除无关干扰因素")

if model:
    # 录入工艺参数卡片
    with st.expander("🛠 录入工艺参数", expanded=True):
        c1, c2, c3 = st.columns(3)
        v_bg = c1.number_input("包高（cm）", value=8.50, format="%.2f")
        v_zj = c1.number_input("粘胶配比%", value=0.50, format="%.2f")
        v_mx = c1.number_input("每箱数量", value=5000)

        v_jx = c2.number_input("机型（cm）", value=4.80, format="%.2f")
        v_dl = c2.number_input("涤纶配比%", value=0.50, format="%.2f")
        v_cn = c2.number_input("小时产能", value=86331)

        v_kz = c3.number_input("克重g/㎡", value=30.00, format="%.2f")
        v_kl = c3.number_input("开料cm", value=9.20, format="%.2f")
        v_rs = c3.number_input("岗位人数", value=4.60, format="%.2f")

        st.markdown("---")
        b1, b2, b_spacer = st.columns([1, 1, 3])
        with b1:
            predict_btn = st.button("开始智能核价", type="primary")
        with b2:
            st.markdown('<div class="btn-reset">', unsafe_allow_html=True)
            if st.button("重置所有参数"):
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        # --- 核心预测逻辑修改 ---
        # 1. 组装当前用户输入的数据
        user_input_dict = {
            "包高（cm）": v_bg, "机型（cm）": v_jx, "克重g/㎡": v_kz,
            "粘胶配比%": v_zj, "涤纶配比%": v_dl, "开料cm": v_kl,
            "每箱数量": v_mx, "小时产能": v_cn, "岗位人数": v_rs
        }

        # 2. 优先查表逻辑：在 raw_df 中寻找完全匹配的记录
        match = raw_df[
            (raw_df['包高（cm）'] == v_bg) &
            (raw_df['机型（cm）'] == v_jx) &
            (raw_df['克重g/㎡'] == v_kz) &
            (raw_df['粘胶配比%'] == v_zj) &
            (raw_df['涤纶配比%'] == v_dl) &
            (raw_df['开料cm'] == v_kl) &
            (raw_df['每箱数量'] == v_mx) &
            (raw_df['小时产能'] == v_cn) &
            (raw_df['岗位人数'] == v_rs)
            ]

        if not match.empty:
            # A. 匹配成功，直接取历史真实值
            final_price = match['成本单价PCS'].values[0]
            st.success("✨ 匹配到历史完全一致订单，已为您精准还原历史实际单价。")
        else:
            # B. 匹配失败，执行 Lasso 智能预测
            input_df = pd.DataFrame([user_input_dict])
            input_scaled = scaler.transform(input_df)
            final_price = max(0, model.predict(input_scaled)[0])
            st.info("💡 历史库中未见完全一致配置，当前为智能预测结果。")

        # --- 结果展示部分 ---
        st.markdown('<div class="section-header">📋 核价结果分析</div>', unsafe_allow_html=True)
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown(f"""
                <div class="price-box">
                    <p style='color:#666; margin:0; font-size:14px;'>预估成本单价 (PCS)</p>
                    <h1 style='color:#d9534f; margin:10px 0; border:none; font-size:32px;'>¥ {final_price:.6f}</h1>
                </div>
            """, unsafe_allow_html=True)

            avg_price = raw_df['成本单价PCS'].mean()
            delta = ((final_price - avg_price) / avg_price) * 100
            st.markdown(f"""
                <div class="delta-box">
                    <p style='margin:0; color:#004085; font-size:15px;'>较历史均值偏差: 
                    <span style='font-weight:bold; color:{"#d9534f" if delta > 0 else "#28a745"};'>{delta:+.2f}%</span></p>
                </div>
            """, unsafe_allow_html=True)

            if delta <= 10:
                st.markdown(
                    '<div class="judgment-box judgment-success"><span>✅ 当前成本控制在合理范围内，可正常报价。</span></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="judgment-box judgment-warning"><span>⚠️ 当前配置成本偏高，建议优化工艺参数或核对物料。</span></div>',
                    unsafe_allow_html=True)

        with res_col2:
            coef_df = pd.DataFrame({'特征': all_feature_names, '影响权重': model.coef_}).sort_values(by='影响权重',
                                                                                                     ascending=True)
            fig_coef = px.bar(coef_df, x='影响权重', y='特征', orientation='h',
                              title="影响权重分析",
                              color='影响权重', color_continuous_scale='RdYlGn_r')

            fig_coef.update_layout(
                height=450,
                title={'text': "各工艺参数对成本的影响权重", 'x': 0.5, 'xanchor': 'center'},
                margin=dict(l=20, r=20, t=60, b=20),
                xaxis=dict(tickformat=".6f"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_coef, use_container_width=True)

        # 历史基准对照分析
        st.markdown("---")
        st.markdown('<div class="section-header">📊 历史基准对照分析</div>', unsafe_allow_html=True)
        chart_col, text_col = st.columns([2.5, 1])

        with chart_col:
            from scipy.stats import gaussian_kde

            hist_data = raw_df['成本单价PCS'].dropna()
            kde = gaussian_kde(hist_data)
            x_range = np.linspace(hist_data.min() * 0.8, hist_data.max() * 1.2, 200)
            y_kde = kde(x_range)

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Scatter(
                x=x_range, y=y_kde,
                fill='tozeroy',
                line=dict(color='#abc1d1', width=3),
                fillcolor='rgba(171, 193, 209, 0.4)',
                name='历史成本密度'
            ))

            fig_dist.add_vline(x=final_price, line_width=3, line_dash="dash", line_color="#d9534f")
            fig_dist.add_annotation(
                x=final_price, y=max(y_kde),
                text="当前核价位置",
                showarrow=True, arrowhead=2, arrowcolor="#d9534f",
                font=dict(color="#d9534f", size=12), bgcolor="white"
            )

            fig_dist.update_layout(
                title={'text': "内部成本分布密度图", 'x': 0.5, 'xanchor': 'center'},
                height=400,
                xaxis_title="成本单价 (元/PCS)",
                yaxis_title="出现频率 (密度)",
                plot_bgcolor='white',
                margin=dict(l=20, r=20, t=60, b=20),
                showlegend=False
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with text_col:
            st.markdown(f"""
                <div style="background-color: #f1f3f5; padding: 20px; border-radius: 10px; border: 1px solid #dee2e6; height: 400px;">
                    <h4 style="color: #004085; margin-top: 0;">📈 图像说明</h4>
                    <p style="font-size: 14px; color: #495057; line-height: 1.6;">
                        该图展示了当前报价在公司<b>历史报价库</b>中的相对位置。
                    </p>
                    <ul style="font-size: 13px; color: #495057; padding-left: 15px;">
                        <li><b>蓝色阴影：</b>代表历史订单的成本分布。</li>
                        <li><b>红色虚线：</b>代表您当前的核价结果。</li>
                    </ul>
                    <hr style="margin: 15px 0; border: 0; border-top: 1px solid #dee2e6;">
                    <h5 style="color: #004085;">💡 评价：</h5>
                    <p style="font-size: 13px; color: #495057;">
                        {"<span style='color: #28a745; font-weight: bold;'>[低价位]</span> 极具市场竞争力。" if final_price < hist_data.median() else
            "<span style='color: #fd7e14; font-weight: bold;'>[高价位]</span> 成本偏高，需核查工艺。"}
                    </p>
                </div>
            """, unsafe_allow_html=True)
else:
    st.error("模型未就绪，请检查 Excel 数据源。")
