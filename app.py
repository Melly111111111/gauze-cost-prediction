import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 设置中文字体，防止图表乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(page_title="定制产品成本估算系统", layout="wide")


# ==========================================
# 1. 后端逻辑：模型训练与标准化处理 (带缓存)
# ==========================================
@st.cache_resource(show_spinner="正在从 Excel 数据训练 Lasso 模型...")
def train_and_cache_model():
    # 这里修改为你的 Excel 文件名
    file_path = '纱布AI报价数据1.xlsx'

    try:
        # 使用 pd.read_excel 读取 Excel 文件
        # 如果有多个 Sheet，可以用 sheet_name='Sheet1' 指定
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        st.error(f"❌ 找不到 Excel 文件：{file_path}，请确保文件在当前目录下。")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"❌ 读取 Excel 出错: {e}")
        return None, None, None, None, None

    # 定义目标变量和特征变量
    target_col = '成本单价PCS'

    # 确保 Excel 中包含这些特征
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    feature_names = X.columns.tolist()

    # --- 核心步骤：标准化 ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 核心步骤：构建 Lasso 模型 ---
    loo = LeaveOneOut()
    model = LassoCV(alphas=np.logspace(-6, 2, 200), cv=loo, max_iter=50000)
    model.fit(X_scaled, y)

    # 计算评估指标
    y_pred = model.predict(X_scaled)
    metrics = {
        'r2': r2_score(y, y_pred),
        'mae': mean_absolute_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'alpha': model.alpha_
    }

    plot_data = {'y': y, 'y_pred': y_pred, 'coef': model.coef_, 'feature_names': feature_names}

    return model, scaler, feature_names, metrics, plot_data


# 初始化加载
model, scaler, feature_cols, metrics, plot_data = train_and_cache_model()

# ==========================================
# 2. 前端 UI 设计
# ==========================================
st.title("纱布产品成本预测系统")
st.markdown("该系统自动分析工艺参数，并实时预测 **成本单价 (PCS)**。")
st.divider()

if model is not None:
    # 布局：左侧输入，右侧显示结果
    main_col, side_col = st.columns([2, 1])

    with main_col:
        with st.form("input_form"):
            st.subheader("📋 工艺与生产参数录入")
            c1, c2, c3 = st.columns(3)

            with c1:
                v1 = st.number_input("包高（cm）", value=10.0)
                v4 = st.number_input("粘胶配比%", value=70.0)
                v7 = st.number_input("每箱数量", value=1000)

            with c2:
                v2 = st.number_input("机型（cm）", value=15.0)
                v5 = st.number_input("涤纶配比%", value=30.0)
                v8 = st.number_input("小时产能", value=5000)

            with c3:
                v3 = st.number_input("克重g/㎡", value=50.0)
                v6 = st.number_input("开料cm", value=20.0)
                v9 = st.number_input("岗位人数", value=4.60, format="%.2f", step=0.01)

            predict_btn = st.form_submit_button("立即计算预测成本", use_container_width=True, type="primary")

    # ==========================================
    # 3. 预测逻辑处理
    # ==========================================
    if predict_btn:
        # 整理输入数据
        input_data = pd.DataFrame([[v1, v2, v3, v4, v5, v6, v7, v8, v9]], columns=feature_cols)

        # 使用 Scaler 标准化输入
        input_scaled = scaler.transform(input_data)

        # 模型预测
        prediction = model.predict(input_scaled)[0]

        with side_col:
            st.success("计算完成")
            # 这里的价格保留4位或6位小数，方便观察微小成本变动
            st.metric(label="预测成本单价 (PCS)", value=f"¥ {max(0, prediction):.6f}")
            st.info("预测结果已经得出。")

    # ==========================================
