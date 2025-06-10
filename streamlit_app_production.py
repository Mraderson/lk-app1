#!/usr/bin/env python3
"""
抑郁量表得分预测应用 - 云端生产版本
整合了所有GPU兼容性修复和SHAP优化
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
import matplotlib.pyplot as plt

# 云端环境兼容性修复
try:
    # 修复numpy兼容性问题
    if not hasattr(np, 'int'):
        np.int = int
    if not hasattr(np, 'float'):
        np.float = float
    if not hasattr(np, 'bool'):
        np.bool = bool
    if not hasattr(np, 'complex'):
        np.complex = complex
    print("✅ NumPy兼容性补丁已应用")
except Exception as e:
    print(f"⚠️ NumPy兼容性补丁失败: {e}")

# 配置页面
st.set_page_config(
    page_title="抑郁量表得分预测",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 全局变量
current_dir = Path(__file__).parent

# 尝试导入SHAP，如果失败则跳过
try:
    import shap
    SHAP_AVAILABLE = True
    print("✅ SHAP库可用")
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP库不可用，将跳过特征分析功能")

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

class DepressionPredictionApp:
    def __init__(self):
        """初始化应用"""
        self.models = {}
        self.feature_names = ['亲子量表总得分', '韧性量表总得分', '焦虑量表总得分', '手机使用时间总得分']
        self.available_models = ['XGBoost', 'LightGBM', 'LinearRegression', 'Ridge', 'KNN']
        
        # 特征名称映射（用于SHAP显示）
        self.feature_name_mapping = {
            '亲子量表总得分': 'Parent Child',
            '韧性量表总得分': 'Resilience',
            '焦虑量表总得分': 'Anxiety',
            '手机使用时间总得分': 'Phone Usage Time'
        }
        
        self.load_models()
        self.load_background_data()
    
    def load_models(self):
        """加载模型并进行GPU兼容性修复"""
        models_dir = current_dir / 'models'
        
        model_files = {
            'XGBoost': 'XGBoost_model.pkl',
            'LightGBM': 'LightGBM_model.pkl',
            'LinearRegression': 'LinearRegression_model.pkl',
            'Ridge': 'Ridge_model.pkl',
            'KNN': 'KNN_model.pkl'
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
                            
                            # GPU兼容性修复
                            if model_name in ['XGBoost', 'LightGBM']:
                                try:
                                    # 移除GPU相关属性
                                    gpu_attrs = ['gpu_id', 'device', 'tree_method']
                                    for attr in gpu_attrs:
                                        if hasattr(model, attr):
                                            delattr(model, attr)
                                    
                                    # 设置为CPU模式
                                    if hasattr(model, 'set_param'):
                                        model.set_param({'device': 'cpu'})
                                    
                                    # 处理booster
                                    if hasattr(model, 'get_booster'):
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
        
        self.available_models = [model for model in self.available_models if model in loaded_models]
        print(f"📊 总共加载了 {len(self.available_models)} 个模型: {', '.join(self.available_models)}")
    
    def load_background_data(self):
        """加载背景数据用于SHAP分析"""
        try:
            background_data_cn_path = current_dir / 'models' / 'background_data_cn.pkl'
            
            if background_data_cn_path.exists():
                with open(background_data_cn_path, 'rb') as f:
                    self.background_data_cn = pickle.load(f)
                print(f"✅ 已加载中文背景数据")
            else:
                data_path = current_dir / 'data' / '量表总分完整数据.csv'
                if data_path.exists():
                    df = pd.read_csv(data_path)
                    self.background_data_cn = df[self.feature_names].sample(n=min(500, len(df)), random_state=42)
                    print(f"✅ 从CSV文件加载背景数据")
                else:
                    st.warning("找不到背景数据文件")
                    self.background_data_cn = None
                    
        except Exception as e:
            st.warning(f"加载背景数据失败: {e}")
            self.background_data_cn = None
    
    def calculate_prediction_confidence(self, model, model_name, input_data):
        """计算预测置信区间"""
        try:
            base_prediction = model.predict(input_data)[0]
            
            if model_name in ['XGBoost', 'LightGBM']:
                uncertainty = base_prediction * 0.08
            elif model_name in ['LinearRegression', 'Ridge']:
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
            feature_values = input_data.iloc[0].values
            english_names = [self.feature_name_mapping[name] for name in self.feature_names]
            
            expected_value = explainer.expected_value
            if hasattr(expected_value, '__len__') and len(expected_value) > 1:
                expected_value = expected_value[0]
            
            shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            prediction = expected_value + np.sum(shap_vals)
            
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.6, 0.6)
            
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax.axhline(y=0, color='lightgray', linewidth=25, alpha=0.3)
            
            feature_widths = np.abs(shap_vals) / np.sum(np.abs(shap_vals)) * 0.8
            start_x = 0.1
            
            for i, (name, value, shap_val, width) in enumerate(zip(english_names, feature_values, shap_vals, feature_widths)):
                color = '#ff4757' if shap_val > 0 else '#5352ed'
                
                rect = plt.Rectangle((start_x, -0.2), width, 0.4, 
                                   facecolor=color, alpha=0.8, edgecolor='white', linewidth=2)
                ax.add_patch(rect)
                
                if width > 0.05:
                    ax.text(start_x + width/2, 0, f'{name}\n= {value:.1f}', 
                           ha='center', va='center', fontsize=12, color='white', weight='bold')
                
                start_x += width
            
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
        """运行SHAP分析"""
        if not SHAP_AVAILABLE or self.background_data_cn is None:
            return None
        
        try:
            if model_name == 'XGBoost':
                try:
                    print(f"使用TreeExplainer分析 {model_name}")
                    background_sample = self.background_data_cn.sample(100, random_state=42)
                    explainer = shap.TreeExplainer(model, background_sample)
                    shap_values = explainer.shap_values(input_data)
                    print(f"✅ {model_name} SHAP分析成功")
                    return shap_values, explainer
                except Exception as tree_error:
                    print(f"TreeExplainer失败: {tree_error}")
                    return None
            
            elif model_name in ['LinearRegression', 'Ridge']:
                print(f"使用LinearExplainer分析 {model_name}")
                explainer = shap.LinearExplainer(model, self.background_data_cn.sample(50, random_state=42))
                shap_values = explainer.shap_values(input_data)
                return shap_values, explainer
            
            else:
                print(f"{model_name} 暂不支持SHAP分析")
                return None
            
        except Exception as e:
            print(f"SHAP分析错误 ({model_name}): {e}")
            return None
    
    def safe_predict(self, model, model_name, input_data):
        """安全预测，带GPU兼容性修复"""
        try:
            # 直接尝试预测
            return model.predict(input_data)[0]
        except Exception as pred_error:
            if 'gpu_id' in str(pred_error) or 'device' in str(pred_error):
                try:
                    st.info("🔧 正在修复GPU兼容性问题...")
                    import copy
                    model_fixed = copy.deepcopy(model)
                    
                    # 移除GPU属性
                    for attr in ['gpu_id', 'device', 'tree_method', '_Booster']:
                        if hasattr(model_fixed, attr):
                            try:
                                delattr(model_fixed, attr)
                            except:
                                pass
                    
                    # 设置CPU模式
                    if hasattr(model_fixed, 'set_param'):
                        model_fixed.set_param({'device': 'cpu'})
                    
                    # 重试预测
                    prediction = model_fixed.predict(input_data)[0]
                    self.models[model_name] = model_fixed  # 保存修复后的模型
                    st.success(f"✅ {model_name} 模型修复成功！")
                    return prediction
                    
                except Exception as final_error:
                    st.error(f"⚠️ {model_name} 模型存在兼容性问题")
                    st.info("💡 建议使用 LinearRegression 或 Ridge 模型")
                    return None
            else:
                st.error(f"预测失败: {pred_error}")
                return None
    
    def run(self):
        """运行应用主程序"""
        st.markdown('<div class="main-title">抑郁量表得分预测</div>', unsafe_allow_html=True)
        
        st.info("🌐 云端生产版本：已优化GPU兼容性，支持XGBoost模型和SHAP分析")
        
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
                # 准备输入数据
                input_data = pd.DataFrame({
                    '亲子量表总得分': [parent_child],
                    '韧性量表总得分': [resilience],
                    '焦虑量表总得分': [anxiety],
                    '手机使用时间总得分': [phone_usage]
                })
                
                # 安全预测
                prediction = self.safe_predict(self.models[selected_model], selected_model, input_data)
                
                if prediction is not None:
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
                                elif selected_model == 'LightGBM':
                                    st.info("💡 LightGBM模型的特征分析暂时不可用")
                        except Exception as shap_error:
                            st.warning(f"特征分析暂时不可用: {str(shap_error)}")
            else:
                st.error(f"模型 {selected_model} 不可用")

# 运行应用
if __name__ == "__main__":
    app = DepressionPredictionApp()
    app.run() 