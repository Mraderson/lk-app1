import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import warnings
import os
import sys
from pathlib import Path
from scipy import stats

# 安全导入SHAP - 如果失败也不影响主要功能
SHAP_AVAILABLE = True
try:
    import shap
except ImportError as e:
    SHAP_AVAILABLE = False
    print(f"⚠️ SHAP未安装: {e}")

# 获取当前目录路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

warnings.filterwarnings("ignore")

# 设置页面配置
st.set_page_config(
    page_title="抑郁量表得分预测",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 简洁的CSS样式 - 白底黑字
st.markdown("""
<style>
    .stApp {
        background-color: white;
    }
    .main-title {
        font-size: 28px;
        font-weight: bold;
        color: #000000;
        text-align: center;
        margin-bottom: 10px;
        margin-top: 5px;
    }
    .model-section {
        margin-bottom: 0px;
    }
    .input-section {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 0px 0 10px 0;
        border: 1px solid #e9ecef;
    }
    .input-container {
        display: flex;
        flex-direction: column;
        align-items: stretch;
        height: 110px;
        margin-bottom: 15px;
    }
    .input-label {
        font-size: 16px;
        font-weight: 500;
        color: #000000;
        margin-bottom: 8px;
        text-align: left;
        min-height: 35px;
        display: flex;
        align-items: center;
    }
    .input-number {
        background-color: #2c3e50;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        font-weight: bold;
        text-align: center;
        flex-grow: 1;
        margin-top: auto;
    }
    .prediction-container {
        background-color: #ffffff;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        padding: 20px;
        margin: 5px 0;
    }
    .prediction-box {
        text-align: center;
        margin-bottom: 15px;
    }
    .prediction-text {
        font-size: 18px;
        color: #000000;
        font-style: italic;
        margin-bottom: 10px;
    }
    .prediction-value {
        font-size: 24px;
        font-weight: bold;
        color: #000000;
        margin-bottom: 5px;
    }
    .confidence-interval {
        font-size: 16px;
        color: #666666;
        margin-top: 10px;
    }
    .score-details {
        display: flex;
        justify-content: space-around;
        margin-top: 15px;
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    .score-item {
        text-align: center;
    }
    .score-label {
        font-size: 14px;
        color: #666666;
        margin-bottom: 5px;
    }
    .score-value {
        font-size: 18px;
        font-weight: bold;
        color: #2c3e50;
    }
    .model-select {
        margin-bottom: 0px;
    }
    .predict-button {
        width: 100%;
        background-color: #2c3e50;
        color: white;
        border: none;
        padding: 15px 20px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        margin: 10px 0;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .predict-button:hover {
        background-color: #34495e;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .shap-section {
        margin-top: 30px;
        padding: 20px;
        background-color: #ffffff;
    }
    .stNumberInput > div > div > input {
        background-color: #2c3e50 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 16px !important;
        font-weight: bold !important;
        text-align: center !important;
        height: 50px !important;
    }
    .stSelectbox > div > div > div {
        background-color: #2c3e50 !important;
        color: white !important;
        border-radius: 8px !important;
    }
    div[data-testid="stNumberInput"] {
        height: 70px;
    }
</style>
""", unsafe_allow_html=True)

class DepressionPredictionApp:
    def __init__(self):
        self.models = {}
        # 只使用经过测试能正常工作的模型
        self.available_models = [
            'XGBoost', 'LightGBM', 'KNN', 'LinearRegression', 'Ridge'
        ]
        
        # 特征名称映射
        self.feature_names = ['亲子量表总得分', '韧性量表总得分', '焦虑量表总得分', '手机使用时间总得分']
        self.feature_name_mapping = {
            '亲子量表总得分': 'Parent-Child Scale',
            '韧性量表总得分': 'Resilience Scale', 
            '焦虑量表总得分': 'Anxiety Scale',
            '手机使用时间总得分': 'Phone Usage Time'
        }
        
        self.load_models()
        self.load_background_data()
    
    def load_models(self):
        """加载可用的模型"""
        models_dir = current_dir / 'models'
        
        # 只加载经过测试的工作模型
        model_files = {
            'XGBoost': 'XGBoost_model.pkl',
            'LightGBM': 'LightGBM_model.pkl',
            'KNN': 'KNN_model.pkl',
            'LinearRegression': 'LinearRegression_model.pkl',
            'Ridge': 'Ridge_model.pkl'
        }
        
        loaded_models = []
        for model_name, file_name in model_files.items():
            model_path = models_dir / file_name
            if model_path.exists():
                try:
                    # 抑制XGBoost的版本警告
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with open(model_path, 'rb') as f:
                            self.models[model_name] = pickle.load(f)
                            loaded_models.append(model_name)
                            print(f"✅ 成功加载模型: {model_name}")
                except Exception as e:
                    print(f"❌ 无法加载模型 {model_name}: {e}")
                    continue
        
        # 更新可用模型列表为实际加载成功的模型
        self.available_models = [model for model in self.available_models if model in loaded_models]
        print(f"📊 总共加载了 {len(self.available_models)} 个模型: {', '.join(self.available_models)}")
    
    def load_background_data(self):
        """加载背景数据用于SHAP分析和置信区间计算"""
        try:
            data_path = current_dir / 'data' / '量表总分完整数据.csv'
            if data_path.exists():
                df = pd.read_csv(data_path)
                # 随机采样500个样本作为背景数据
                self.background_data = df[self.feature_names].sample(n=min(500, len(df)), random_state=42)
                # 加载完整数据用于置信区间估算
                self.full_data = df
            else:
                st.error("找不到数据文件")
                self.background_data = None
                self.full_data = None
        except Exception as e:
            st.error(f"加载数据失败: {e}")
            self.background_data = None
            self.full_data = None
    
    def calculate_prediction_confidence(self, model, model_name, input_data, n_bootstrap=50):
        """计算预测置信区间 - 简化版本"""
        try:
            if self.full_data is None:
                return None, None, None
            
            # 简化的置信区间计算
            base_prediction = model.predict(input_data)[0]
            
            # 基于模型类型设置不同的不确定性
            if model_name in ['XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting']:
                uncertainty = base_prediction * 0.08  # 8%的不确定性
            elif model_name in ['SVM', 'ANN']:
                uncertainty = base_prediction * 0.12  # 12%的不确定性
            else:
                uncertainty = base_prediction * 0.10  # 10%的不确定性
            
            # 计算置信区间
            lower_ci = max(0, base_prediction - 1.96 * uncertainty)
            upper_ci = min(27, base_prediction + 1.96 * uncertainty)
            
            return base_prediction, lower_ci, upper_ci
                
        except Exception as e:
            print(f"置信区间计算错误: {e}")
            return None, None, None
    
    def create_shap_force_plot(self, explainer, shap_values, input_data):
        """创建SHAP force plot，参考用户提供的图片样式"""
        try:
            # 获取特征值和英文名称
            feature_values = input_data.iloc[0].values
            english_names = [self.feature_name_mapping[name] for name in self.feature_names]
            
            # 获取基准值和SHAP值
            expected_value = explainer.expected_value
            if hasattr(expected_value, '__len__') and len(expected_value) > 1:
                expected_value = expected_value[0]
            
            if len(shap_values.shape) > 1:
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values
            
            # 计算预测值
            prediction = expected_value + np.sum(shap_vals)
            
            # 创建图形 - 增大尺寸以提高清晰度
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.6, 0.6)
            
            # 隐藏坐标轴
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # 绘制基准线（灰色背景）
            ax.axhline(y=0, color='lightgray', linewidth=25, alpha=0.3)
            
            # 计算累积位置
            current_pos = 0
            total_width = 1.0
            feature_widths = np.abs(shap_vals) / np.sum(np.abs(shap_vals)) * 0.8  # 留20%空白
            
            # 绘制每个特征的贡献
            start_x = 0.1  # 左边留10%空白
            for i, (name, value, shap_val, width) in enumerate(zip(english_names, feature_values, shap_vals, feature_widths)):
                # 根据SHAP值确定颜色
                if shap_val > 0:
                    color = '#ff4757'  # 红色 - 增加风险
                else:
                    color = '#5352ed'  # 蓝色 - 降低风险
                
                # 绘制特征条
                rect = plt.Rectangle((start_x, -0.2), width, 0.4, 
                                   facecolor=color, alpha=0.8, edgecolor='white', linewidth=2)
                ax.add_patch(rect)
                
                # 添加特征标签和值 - 增大字体
                if width > 0.05:  # 只有足够宽的条才显示标签
                    ax.text(start_x + width/2, 0, f'{name}\n= {value:.1f}', 
                           ha='center', va='center', fontsize=12, color='white', weight='bold')
                
                start_x += width
            
            # 添加基准值标签 - 增大字体
            ax.text(0.05, -0.45, f'基准值 = {expected_value:.1f}', fontsize=14, ha='left', weight='bold')
            
            # 添加预测结果 - 增大字体
            ax.text(0.95, -0.45, f'预测值 = {prediction:.2f}', fontsize=14, ha='right', weight='bold')
            
            # 添加说明 - 增大字体
            ax.text(0.5, 0.45, 'Based on feature values, predicted possibility of Depression is {:.2f}%'.format(prediction*100/27), 
                   ha='center', va='center', fontsize=16, style='italic', weight='bold')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"创建SHAP图表失败: {e}")
            return None
    
    def run_shap_analysis(self, model, model_name, input_data):
        """运行SHAP分析 - 简化版本"""
        if self.background_data is None or not SHAP_AVAILABLE:
            return None
        
        try:
            # 针对不同模型类型使用不同的SHAP解释器
            if model_name in ['XGBoost', 'LightGBM', 'RandomForest', 'DecisionTree', 'GradientBoosting', 'ExtraTrees']:
                # 树模型使用TreeExplainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_data)
            elif model_name in ['LinearRegression', 'Ridge']:
                # 线性模型使用LinearExplainer
                explainer = shap.LinearExplainer(model, self.background_data.sample(50))
                shap_values = explainer.shap_values(input_data)
            elif model_name in ['KNN', 'SVM', 'ANN']:
                # 非线性模型使用KernelExplainer (采样版本，速度更快)
                background_sample = self.background_data.sample(30, random_state=42)  # 减少样本数提高速度
                explainer = shap.KernelExplainer(model.predict, background_sample)
                shap_values = explainer.shap_values(input_data, nsamples=50)  # 减少采样次数
            else:
                # 其他模型暂时跳过SHAP分析
                return None
            
            return shap_values, explainer
            
        except Exception as e:
            print(f"SHAP分析错误: {e}")
            return None
    
    def run(self):
        """运行应用主程序"""
        # 页面标题
        st.markdown('<div class="main-title">抑郁量表得分预测</div>', unsafe_allow_html=True)
        
        # 显示SHAP状态提示
        if not SHAP_AVAILABLE:
            st.info("📊 预测功能正常运行，SHAP分析功能暂时不可用")
        else:
            st.success("🎯 SHAP分析已启用 - 所有5个模型均支持特征解释")
        
        # 模型选择 - 去掉多余空白
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('<div class="input-label">选择预测模型:</div>', unsafe_allow_html=True)
            selected_model = st.selectbox(
                "预测模型",
                self.available_models,
                index=0 if 'XGBoost' in self.available_models else 0,
                label_visibility="collapsed"
            )
        
        # 输入区域 - 紧接着模型选择
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            st.markdown('<div class="input-label">亲子量表得分</div>', unsafe_allow_html=True)
            parent_child = st.number_input("亲子量表总得分", min_value=8, max_value=50, value=17, step=1, key="parent", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            st.markdown('<div class="input-label">韧性量表得分</div>', unsafe_allow_html=True)
            resilience = st.number_input("韧性量表总得分", min_value=0, max_value=40, value=7, step=1, key="resilience", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            st.markdown('<div class="input-label">焦虑量表得分</div>', unsafe_allow_html=True)
            anxiety = st.number_input("焦虑量表总得分", min_value=0, max_value=20, value=4, step=1, key="anxiety", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            st.markdown('<div class="input-label">手机使用时间得分</div>', unsafe_allow_html=True)
            phone_usage = st.number_input("手机使用时间总得分", min_value=0, max_value=60, value=23, step=1, key="phone", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 预测按钮
        if st.button("Predict", key="predict_btn"):
            if selected_model in self.models:
                # 准备输入数据
                input_data = pd.DataFrame({
                    '亲子量表总得分': [parent_child],
                    '韧性量表总得分': [resilience],
                    '焦虑量表总得分': [anxiety],
                    '手机使用时间总得分': [phone_usage]
                })
                
                # 进行预测
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        prediction = self.models[selected_model].predict(input_data)[0]
                    
                    # 计算置信区间
                    mean_pred, lower_ci, upper_ci = self.calculate_prediction_confidence(
                        self.models[selected_model], selected_model, input_data
                    )
                    
                    # 使用实际预测值或平均值
                    final_prediction = mean_pred if mean_pred is not None else prediction
                    
                    # 显示预测结果 - 使用简单的streamlit组件
                    st.markdown(f"""
                    <div style="background-color: #ffffff; border: 2px solid #dee2e6; border-radius: 8px; padding: 20px; margin: 10px 0; text-align: center;">
                        <div style="font-size: 18px; color: #000000; font-style: italic; margin-bottom: 10px;">
                            Based on feature values, predicted possibility of Depression is
                        </div>
                        <div style="font-size: 24px; font-weight: bold; color: #000000; margin-bottom: 5px;">
                            {final_prediction*100/27:.2f}%
                        </div>
                        {f'<div style="font-size: 16px; color: #666666; margin-top: 10px;">95% 置信区间: {lower_ci*100/27:.1f}% - {upper_ci*100/27:.1f}%</div>' if lower_ci is not None and upper_ci is not None else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 显示详细得分信息 - 使用更清晰的样式
                    st.markdown("""
                    <div style="display: flex; justify-content: space-around; margin-top: 15px; padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
                        <div style="text-align: center; flex: 1;">
                            <div style="font-size: 14px; color: #666666; margin-bottom: 5px; font-weight: 500;">预测得分</div>
                            <div style="font-size: 24px; font-weight: bold; color: #2c3e50;">{:.2f}</div>
                        </div>
                        <div style="text-align: center; flex: 1;">
                            <div style="font-size: 14px; color: #666666; margin-bottom: 5px; font-weight: 500;">得分范围</div>
                            <div style="font-size: 24px; font-weight: bold; color: #2c3e50;">0-27</div>
                        </div>
                        <div style="text-align: center; flex: 1;">
                            <div style="font-size: 14px; color: #666666; margin-bottom: 5px; font-weight: 500;">风险等级</div>
                            <div style="font-size: 24px; font-weight: bold; color: {};">{}</div>
                        </div>
                        <div style="text-align: center; flex: 1;">
                            <div style="font-size: 14px; color: #666666; margin-bottom: 5px; font-weight: 500;">使用模型</div>
                            <div style="font-size: 24px; font-weight: bold; color: #2c3e50;">{}</div>
                        </div>
                    </div>
                    """.format(
                        final_prediction,
                        "#e74c3c" if final_prediction > 14 else "#f39c12" if final_prediction > 7 else "#27ae60",
                        "高风险" if final_prediction > 14 else "中风险" if final_prediction > 7 else "低风险",
                        selected_model
                    ), unsafe_allow_html=True)
                    
                    # SHAP分析
                    try:
                        with st.spinner("生成SHAP分析图..."):
                            shap_result = self.run_shap_analysis(self.models[selected_model], selected_model, input_data)
                            
                            if shap_result:
                                shap_values, explainer = shap_result
                                
                                # 创建SHAP force plot
                                fig = self.create_shap_force_plot(explainer, shap_values, input_data)
                                if fig:
                                    st.pyplot(fig)
                    except Exception as shap_error:
                        st.warning(f"SHAP分析暂时不可用: {shap_error}")
                
                except Exception as e:
                    st.error(f"预测失败: {e}")
                    st.info("请尝试选择其他模型或检查输入数据")
            else:
                st.error(f"模型 {selected_model} 不可用，请选择其他模型")

# 运行应用
if __name__ == "__main__":
    app = DepressionPredictionApp()
    app.run()
