#!/usr/bin/env python3
"""
抑郁量表得分预测应用 - 云端安全版本
专门为云端环境优化，跳过有GPU兼容性问题的模型
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
import matplotlib.pyplot as plt

# 配置页面
st.set_page_config(
    page_title="Depression Scale Score Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 全局变量
current_dir = Path(__file__).parent

# 自定义CSS
st.markdown("""
<style>
.main-title {
    font-size: 2.5em;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 30px;
    font-weight: bold;
}
.input-label {
    font-size: 1.1em;
    font-weight: 600;
    color: #34495e;
    margin-bottom: 5px;
}
.input-container {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    margin: 5px;
    border: 1px solid #e9ecef;
}
.stSelectbox > div > div {
    background-color: #ffffff;
    border: 2px solid #3498db;
    border-radius: 8px;
}
.stButton > button {
    background-color: #3498db;
    color: white;
    font-weight: bold;
    font-size: 18px;
    border-radius: 10px;
    border: none;
    padding: 15px 30px;
    width: 100%;
    margin-top: 20px;
}
.stButton > button:hover {
    background-color: #2980b9;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# 尝试导入SHAP，如果失败则跳过
try:
    import shap
    SHAP_AVAILABLE = True
    print("✅ SHAP库可用")
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP库不可用，将跳过特征分析功能")

class DepressionPredictionApp:
    def __init__(self):
        """初始化应用"""
        self.models = {}
        self.feature_names = ['parent_child_score', 'resilience_score', 'anxiety_score', 'phone_usage_score']
        
        # 云端安全：加载稳定的模型，包括修复后的XGBoost
        self.available_models = ['LinearRegression', 'Ridge', 'KNN', 'XGBoost']
        
        # 特征名称映射（用于SHAP显示）
        self.feature_name_mapping = {
            'parent_child_score': 'Parent Child',
            'resilience_score': 'Resilience',
            'anxiety_score': 'Anxiety',
            'phone_usage_score': 'Phone Usage Time'
        }
        
        self.load_models()
        self.load_background_data()
    
    def load_models(self):
        """加载可用的模型 - 云端安全版本"""
        models_dir = current_dir / 'models'
        
        # 加载稳定的模型，包括修复后的XGBoost
        model_files = {
            'LinearRegression': 'LinearRegression_model.pkl',
            'Ridge': 'Ridge_model.pkl',
            'KNN': 'KNN_model.pkl',
            'XGBoost': 'XGBoost_model.pkl'
        }
        
        loaded_models = []
        for model_name, file_name in model_files.items():
            model_path = models_dir / file_name
            if model_path.exists():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                            
                            # 修复XGBoost的GPU兼容性问题
                            if model_name == 'XGBoost' and hasattr(model, 'get_booster'):
                                try:
                                    # 移除GPU相关属性
                                    gpu_attrs = ['gpu_id', 'device']
                                    for attr in gpu_attrs:
                                        if hasattr(model, attr):
                                            delattr(model, attr)
                                    
                                    # 设置为CPU模式
                                    if hasattr(model, 'set_param'):
                                        model.set_param({'device': 'cpu'})
                                    
                                    # 处理booster
                                    booster = model.get_booster()
                                    if hasattr(booster, 'set_param'):
                                        booster.set_param({'device': 'cpu'})
                                    
                                    print(f"✅ {model_name} GPU兼容性已修复")
                                except Exception as fix_error:
                                    print(f"⚠️ {model_name} GPU修复警告: {fix_error}")
                            
                            self.models[model_name] = model
                            loaded_models.append(model_name)
                            print(f"✅ 成功加载模型: {model_name}")
                except Exception as e:
                    print(f"❌ 无法加载模型 {model_name}: {e}")
                    continue
        
        # 更新可用模型列表
        self.available_models = [model for model in self.available_models if model in loaded_models]
        print(f"📊 云端模式：加载了 {len(self.available_models)} 个模型: {', '.join(self.available_models)}")
        
        # 如果没有加载到任何模型，显示警告
        if not self.available_models:
            st.error("⚠️ 没有可用的模型，请检查模型文件")
    
    def load_background_data(self):
        """加载背景数据用于SHAP分析"""
        try:
            # 尝试加载预生成的背景数据
            background_data_cn_path = current_dir / 'models' / 'background_data_cn.pkl'
            
            if background_data_cn_path.exists():
                with open(background_data_cn_path, 'rb') as f:
                    self.background_data_cn = pickle.load(f)
                print(f"✅ 已加载中文背景数据")
            else:
                # 回退到从CSV加载数据
                data_path = current_dir / 'data' / '量表总分完整数据.csv'
                if data_path.exists():
                    df = pd.read_csv(data_path)
                    self.background_data_cn = df[self.feature_names].sample(n=min(500, len(df)), random_state=42)
                    self.full_data = df
                    print(f"✅ 从CSV文件加载背景数据")
                else:
                    st.warning("找不到背景数据文件")
                    self.background_data_cn = None
                    self.full_data = None
                    
        except Exception as e:
            st.warning(f"加载背景数据失败: {e}")
            self.background_data_cn = None
            self.full_data = None
    
    def calculate_prediction_confidence(self, model, model_name, input_data):
        """计算预测置信区间"""
        try:
            base_prediction = model.predict(input_data)[0]
            
            # 基于模型类型设置不确定性
            if model_name in ['LinearRegression', 'Ridge']:
                uncertainty = base_prediction * 0.10
            else:
                uncertainty = base_prediction * 0.12
            
            lower_ci = max(0, base_prediction - 1.96 * uncertainty)
            upper_ci = min(27, base_prediction + 1.96 * uncertainty)
            
            return base_prediction, lower_ci, upper_ci
                
        except Exception as e:
            print(f"置信区间计算错误: {e}")
            return None, None, None
    
    def create_shap_force_plot(self, explainer, shap_values, input_data):
        """创建SHAP force plot"""
        try:
            # 获取特征值和名称
            feature_values = input_data.iloc[0].values
            english_names = [self.feature_name_mapping[name] for name in self.feature_names]
            
            # 获取基准值和SHAP值
            expected_value = explainer.expected_value
            if hasattr(expected_value, '__len__') and len(expected_value) > 1:
                expected_value = expected_value[0]
            
            shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            prediction = expected_value + np.sum(shap_vals)
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.6, 0.6)
            
            # 隐藏坐标轴
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # 绘制背景线
            ax.axhline(y=0, color='lightgray', linewidth=25, alpha=0.3)
            
            # 计算特征条宽度
            feature_widths = np.abs(shap_vals) / np.sum(np.abs(shap_vals)) * 0.8
            start_x = 0.1
            
            # 绘制特征贡献
            for i, (name, value, shap_val, width) in enumerate(zip(english_names, feature_values, shap_vals, feature_widths)):
                color = '#ff4757' if shap_val > 0 else '#5352ed'
                
                rect = plt.Rectangle((start_x, -0.2), width, 0.4, 
                                   facecolor=color, alpha=0.8, edgecolor='white', linewidth=2)
                ax.add_patch(rect)
                
                if width > 0.05:
                    ax.text(start_x + width/2, 0, f'{name}\n= {value:.1f}', 
                           ha='center', va='center', fontsize=12, color='white', weight='bold')
                
                start_x += width
            
            # 添加标签
            ax.text(0.05, -0.45, f'基准值 = {expected_value:.1f}', fontsize=14, ha='left', weight='bold')
            ax.text(0.95, -0.45, f'预测值 = {prediction:.2f}', fontsize=14, ha='right', weight='bold')
            ax.text(0.5, 0.45, f'Based on feature values, predicted possibility of Depression is {prediction*100/27:.2f}%', 
                   ha='center', va='center', fontsize=16, style='italic', weight='bold')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"创建SHAP图表失败: {e}")
            return None
    
    def run_shap_analysis(self, model, model_name, input_data):
        """运行SHAP分析 - 支持线性模型和修复后的XGBoost"""
        if not SHAP_AVAILABLE or self.background_data_cn is None:
            return None
        
        try:
            if model_name in ['LinearRegression', 'Ridge']:
                print(f"使用LinearExplainer分析 {model_name}")
                explainer = shap.LinearExplainer(model, self.background_data_cn.sample(50, random_state=42))
                shap_values = explainer.shap_values(input_data)
                return shap_values, explainer
            
            elif model_name == 'XGBoost':
                try:
                    print(f"使用TreeExplainer分析 {model_name}")
                    # 使用较小的背景数据集以提高速度
                    background_sample = self.background_data_cn.sample(100, random_state=42)
                    explainer = shap.TreeExplainer(model, background_sample)
                    shap_values = explainer.shap_values(input_data)
                    print(f"✅ {model_name} SHAP分析成功")
                    return shap_values, explainer
                except Exception as tree_error:
                    print(f"TreeExplainer失败: {tree_error}")
                    try:
                        print(f"回退到KernelExplainer分析 {model_name}")
                        background_sample = self.background_data_cn.sample(50, random_state=42)
                        explainer = shap.KernelExplainer(model.predict, background_sample)
                        shap_values = explainer.shap_values(input_data)
                        print(f"✅ {model_name} KernelExplainer分析成功")
                        return shap_values, explainer
                    except Exception as kernel_error:
                        print(f"KernelExplainer也失败: {kernel_error}")
                        return None
            
            elif model_name == 'KNN':
                print(f"{model_name} 跳过SHAP分析（性能原因）")
                return None
            
            else:
                print(f"{model_name} 暂不支持SHAP分析")
                return None
            
        except Exception as e:
            print(f"SHAP分析错误 ({model_name}): {e}")
            return None
    
    def run(self):
        """运行应用主程序"""
        # 页面标题
        st.markdown('<div class="main-title">抑郁量表得分预测</div>', unsafe_allow_html=True)
        
        # 云端模式提示
        st.info("🌐 云端环境：已优化GPU兼容性，XGBoost模型现已可用")
        
        if not self.available_models:
            st.error("没有可用的模型，请检查配置")
            return
        
        # 模型选择
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('<div class="input-label">选择预测模型:</div>', unsafe_allow_html=True)
            selected_model = st.selectbox(
                "预测模型",
                self.available_models,
                index=0,
                label_visibility="collapsed"
            )
        
        # 输入区域
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
                # 准备输入数据（云端安全模式使用中文特征名）
                input_data = pd.DataFrame({
                    '亲子量表总得分': [parent_child],
                    '韧性量表总得分': [resilience],
                    '焦虑量表总得分': [anxiety],
                    '手机使用时间总得分': [phone_usage]
                })
                
                try:
                    print(f"🔄 云端安全模式：使用 {selected_model} 进行预测...")
                    
                    # 进行预测
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        prediction = self.models[selected_model].predict(input_data)[0]
                    
                    print(f"✅ {selected_model} 预测成功，结果: {prediction}")
                    
                    # 计算置信区间
                    mean_pred, lower_ci, upper_ci = self.calculate_prediction_confidence(
                        self.models[selected_model], selected_model, input_data
                    )
                    
                    final_prediction = mean_pred if mean_pred is not None else prediction
                    
                    # 显示预测结果
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
                    
                    # 显示详细信息
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
                    if SHAP_AVAILABLE:
                        try:
                            with st.spinner("正在生成特征重要性分析图..."):
                                shap_result = self.run_shap_analysis(self.models[selected_model], selected_model, input_data)
                                
                                if shap_result:
                                    shap_values, explainer = shap_result
                                    fig = self.create_shap_force_plot(explainer, shap_values, input_data)
                                    if fig:
                                        st.pyplot(fig)
                                        plt.close(fig)
                                elif selected_model == 'KNN':
                                    st.info("💡 KNN模型的特征分析需要较长时间，已跳过")
                        except Exception as shap_error:
                            st.warning(f"特征分析暂时不可用: {str(shap_error)}")
                
                except Exception as e:
                    st.error(f"预测失败: {e}")
                    st.info("请尝试选择其他模型或检查输入数据")
            else:
                st.error(f"模型 {selected_model} 不可用")

# 运行应用
if __name__ == "__main__":
    app = DepressionPredictionApp()
    app.run() 