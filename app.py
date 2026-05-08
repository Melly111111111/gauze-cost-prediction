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
        text-align: center; /* 【新增：让文字居中】 */
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
    }

    /* --- 按钮配色对标 --- */
    /* 1. 开始智能核价按钮 - 橙色 (#fd7e14) */
    div.stButton > button[kind="primary"] {
        background-color: #fd7e14 !important;
        border-color: #fd7e14 !important;
        color: white !important;
        width: 100%;
        height: 50px;
        font-weight: bold;
    }

    /* 2. 重置按钮 - 蓝色 (#007bff) */
    .btn-reset button {
        background-color: #007bff !important;
        color: white !important;
        border: none !important;
        white-space: nowrap;
    }

    /* 特征标题背景 */
    .section-header {
        background-color: #e9f1f7;
        color: #004085;
        font-weight: bold;
        padding: 5px 10px;
        border-left: 5px solid #004085;
        margin-bottom: 15px;
    }
    /* 预估单价的卡片 */
    .price-box {
        background-color: #ffffff;
        border: 2px solid #004085;
        border-radius: 10px 10px 0 0; /* 顶部圆角 */
        padding: 20px;
        text-align: center;
        margin-bottom: 25px; /* 【核心修改：增加下外边距，拉开与下方偏差模块的距离】 */
    }

    /* 较历史均值偏差的浅色背景框 */
    .delta-box {
        background-color: #e9f1f7;
        border-radius: 0 0 10px 10px; /* 底部圆角 */
        padding: 15px;
        text-align: center;
        border: 1px solid #dee2e6;
        border-top: none; /* 去掉顶边框，使其视觉上与上方呼应但有间距 */
    }
    /* 成本判断框基础样式 */
.judgment-box {
    padding: 15px;
    border-radius: 8px;
    margin-top: 20px; /* 与上方模块拉开距离 */
    text-align: center;
    font-weight: bold;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
}

/* 合理状态 - 绿色调 */
.judgment-success {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

/* 偏高状态 - 黄色调 */
.judgment-warning {
    background-color: #fff3cd;
    color: #856404;
    border: 1px solid #ffeeba;
}
</style>
""", unsafe_allow_html=True)


# 3. 数据与模型加载 (保持原逻辑)
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
    except:
        return None, None, None, None


model, scaler, all_feature_names, raw_df = train_model()

# --- 界面展示 ---
st.title("纱布产品成本智能决策系统")
st.info("本系统对生产工艺参数进行预测,可以自动识别非线性关系并剔除无关干扰因素")

if model:
    # 参数输入区
    with st.expander("🛠 录入工艺参数", expanded=True):
        c1, c2, c3 = st.columns(3)
        v_bg = c1.number_input("包高（cm）", value=8.50)
        v_zj = c1.number_input("粘胶配比%", value=0.50)
        v_mx = c1.number_input("每箱数量", value=5000)

        v_jx = c2.number_input("机型（cm）", value=4.80)
        v_dl = c2.number_input("涤纶配比%", value=0.50)
        v_cn = c2.number_input("小时产能", value=86331)

        v_kz = c3.number_input("克重g/㎡", value=30.00)
        v_kl = c3.number_input("开料cm", value=9.20)
        v_rs = c3.number_input("岗位人数", value=4.60)

        # 底部按钮区
        st.markdown("---")
        b1, b2, b_spacer = st.columns([1, 1, 3])
        with b1:
            # 按钮通过 type="primary" 触发 CSS 中的橙色样式
            predict_btn = st.button("开始智能核价", type="primary")
        with b2:
            st.markdown('<div class="btn-reset">', unsafe_allow_html=True)
            reset_btn = st.button("重置所有参数")
            if reset_btn: st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        # 预测逻辑
        user_input = {"包高（cm）": v_bg, "机型（cm）": v_jx, "克重g/㎡": v_kz, "粘胶配比%": v_zj,
                      "涤纶配比%": v_dl, "开料cm": v_kl, "每箱数量": v_mx, "小时产能": v_cn, "岗位人数": v_rs}
        full_input_df = pd.DataFrame(0.0, index=[0], columns=all_feature_names)
        for k, v in user_input.items():
            if k in all_feature_names: full_input_df[k] = v

        input_scaled = scaler.transform(full_input_df)
        final_price = max(0, model.predict(input_scaled)[0])

        # 结果分析展示
        st.markdown('<div class="section-header">📋 核价结果分析</div>', unsafe_allow_html=True)
        res_col1, res_r = st.columns([1, 2])

        with res_col1:
            # 1. 预估成本单价模块
            st.markdown(f"""
            <div class="result-card" style="margin-bottom: 25px;">
                <p style='color:#666; margin:0; font-size:14px;'>预估成本单价 (PCS)</p>
                <h1 style='color:#d9534f; margin:10px 0; font-size:32px;'>¥ {final_price:.6f}</h1>
            </div>
            """, unsafe_allow_html=True)

            # 2. 计算并显示较历史均值偏差
            # 确保在 if delta <= 10 之前已经定义了 delta
            avg_price = raw_df['成本单价PCS'].mean()
            delta = ((final_price - avg_price) / avg_price) * 100

            st.markdown(f"""
            <div class="delta-box" style="background-color: #e9f1f7; padding: 15px; border-radius: 8px; text-align: center;">
                <p style='margin:0; color:#004085;'>
                    较历史均值偏差: 
                    <span style='font-weight:bold; color:{"#d9534f" if delta > 0 else "#28a745"};'>
                        {delta:+.2f}%
                    </span>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # 3. 成本判断模块 (此时 delta 已被定义)
            if delta <= 10:
                st.markdown("""
                <div class="judgment-box judgment-success" style="background-color: #d4edda; color: #155724; padding: 15px; border-radius: 8px; margin-top: 20px; text-align: center;">
                    <span>✅ 当前成本控制在合理范围内。</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="judgment-box judgment-warning" style="background-color: #fff3cd; color: #856404; padding: 15px; border-radius: 8px; margin-top: 20px; text-align: center;">
                    <span>⚠️ 当前配置成本偏高，建议优化工艺参数。</span>
                </div>
                """, unsafe_allow_html=True)
        with res_r:
            coef_df = pd.DataFrame({'特征': all_feature_names, '影响权重': model.coef_})
            coef_df = coef_df[coef_df['影响权重'] != 0].sort_values(by='影响权重', ascending=False)

            fig_coef = px.bar(coef_df, x='影响权重', y='特征', orientation='h',
                              title="影响权重分析",  # 简短标题
                              color='影响权重', color_continuous_scale='RdYlGn_r')

            # 修改点：title_x=0.5 使标题居中，height 增加高度
            fig_coef.update_layout(
                height=500,  # 增加高度，原为 300
                title={'text': "影响权重分析", 'x': 0.5, 'xanchor': 'center'},  # 标题居中
                margin=dict(l=20, r=20, t=60, b=20),
                xaxis=dict(tickformat=".6f"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_coef, use_container_width=True)
        # 3. 行业对标分析模块
        st.markdown("---")
        st.markdown('<div class="section-header">📊 历史报价对照分析</div>', unsafe_allow_html=True)

        # 使用分栏布局：左侧放图，右侧放说明文字
        chart_col, text_col = st.columns([2.5, 1])

        with chart_col:
            # 准备平滑曲线数据 (使用 KDE 模拟)
            import numpy as np
            from scipy.stats import gaussian_kde

            # 提取历史价格数据
            hist_data = raw_df['成本单价PCS'].dropna()
            kde = gaussian_kde(hist_data)
            x_range = np.linspace(hist_data.min() * 0.8, hist_data.max() * 1.2, 200)
            y_kde = kde(x_range)

            # 创建更美观的面积图
            fig_dist = go.Figure()

            # 绘制背景填充的密度曲线
            fig_dist.add_trace(go.Scatter(
                x=x_range, y=y_kde,
                fill='tozeroy',
                line=dict(color='#abc1d1', width=3),
                fillcolor='rgba(171, 193, 209, 0.4)',
                name='历史成本密度'
            ))

            # 绘制当前预测值的指示线
            fig_dist.add_vline(
                x=final_price,
                line_width=3,
                line_dash="dash",
                line_color="#d9534f"
            )

            # 添加指示文字锚点
            fig_dist.add_annotation(
                x=final_price, y=max(y_kde),
                text="当前报价位置",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#d9534f",
                font=dict(color="#d9534f", size=12),
                bgcolor="white"
            )

            fig_dist.update_layout(
                title={'text': "成本分布密度图", 'x': 0.5, 'xanchor': 'center'},
                height=450,
                xaxis_title="成本单价 (元/PCS)",
                yaxis_title="出现频率 (密度)",
                plot_bgcolor='white',
                margin=dict(l=20, r=20, t=60, b=20),
                showlegend=False
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with text_col:
            # 右侧文字说明模块
            st.markdown(f"""
                    <div style="background-color: #f1f3f5; padding: 20px; border-radius: 10px; border: 1px solid #dee2e6; height: 450px;">
                        <h4 style="color: #004085; margin-top: 0;">📈 图像说明</h4>
                        <p style="font-size: 14px; color: #495057; line-height: 1.6;">
                            该图展示了当前报价在历史报价库中的<b>相对位置</b>辅助评估本次成本的内部合理性。
                        </p>
                        <ul style="font-size: 13px; color: #495057; padding-left: 15px;">
                            <li><b>蓝色阴影区：</b>代表公司历史订单成本区间。峰值越高，代表该价格越常见。</li>
                            <li><b>红色虚线：</b>代表您当前的计算结果（¥{final_price:.6f}）。</li>
                        </ul>
                        <hr style="margin: 15px 0; border: 0; border-top: 1px solid #dee2e6;">
                        <h5 style="color: #004085;">💡 决策建议：</h5>
                        <p style="font-size: 13px; color: #495057;">
                            {"<span style='color: #28a745; font-weight: bold;'>[优选]</span> 当前成本处于历史低位，具有极强的市场竞争优势。" if final_price < hist_data.median() else
            "<span style='color: #fd7e14; font-weight: bold;'>[预警]</span> 当前成本高于历史中位数，请核查工艺参数（如岗位人数或克重）是否有优化空间。"}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

else:
    st.error("数据加载失败，请检查 Excel 文件。")
