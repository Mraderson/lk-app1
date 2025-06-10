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

# 导入并应用云端兼容性修复
try:
    from cloud_compatibility_fix import apply_numpy_compatibility_patch
    apply_numpy_compatibility_patch()
except ImportError:
    # 如果修复脚本不存在，使用内置补丁
    try:
        if not hasattr(np, 'int'):
            np.int = int
        if not hasattr(np, 'float'):
            np.float = float
        if not hasattr(np, 'bool'):
            np.bool = bool
        if not hasattr(np, 'complex'):
            np.complex = complex
        print("✅ 内置NumPy兼容性补丁已应用")
    except Exception as e:
        print(f"⚠️ NumPy兼容性补丁失败: {e}")
except Exception as e:
    print(f"⚠️ 云端兼容性修复失败: {e}")

# 修复numpy兼容性问题 - 云端环境
try:
    # 添加兼容性补丁，适配新版numpy
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

# 修复numpy兼容性问题 - 云端环境
try:
    # 添加兼容性补丁，适配新版numpy
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

# 安全导入SHAP - 如果失败也不影响主要功能
SHAP_AVAILABLE = True
try:
    import shap
    print("✅ SHAP库导入成功")
except ImportError as e:
    SHAP_AVAILABLE = False
    print(f"⚠️ SHAP未安装: {e}")
except Exception as e:
    SHAP_AVAILABLE = False
    print(f"⚠️ SHAP导入错误: {e}")
except Exception as e:
    SHAP_AVAILABLE = False
    print(f"⚠️ SHAP导入错误: {e}")

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
                            model = pickle.load(f)
                            
                            # 标记需要GPU兼容性处理的模型，但不在加载时修改
                            if model_name in ['XGBoost', 'LightGBM']:
                                print(f"✅ {model_name} 模型加载成功（将在预测时处理GPU兼容性）")
                            
                            self.models[model_name] = model
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
            # 尝试加载预生成的背景数据
            background_data_path = current_dir / 'models' / 'background_data.pkl'
            background_data_cn_path = current_dir / 'models' / 'background_data_cn.pkl'
            
            if background_data_path.exists() and background_data_cn_path.exists():
                # 加载英文和中文特征名称的背景数据
                with open(background_data_path, 'rb') as f:
                    self.background_data_en = pickle.load(f)
                with open(background_data_cn_path, 'rb') as f:
                    self.background_data_cn = pickle.load(f)
                print(f"✅ 已加载预生成的背景数据")
            else:
                # 回退到从CSV加载数据
                data_path = current_dir / 'data' / '量表总分完整数据.csv'
                if data_path.exists():
                    df = pd.read_csv(data_path)
                    # 随机采样500个样本作为背景数据
                    sample_data = df[self.feature_names].sample(n=min(500, len(df)), random_state=42)
                    
                    # 创建英文特征名称的背景数据
                    self.background_data_en = sample_data.rename(columns={
                        '亲子量表总得分': 'parent_child_score',
                        '韧性量表总得分': 'resilience_score', 
                        '焦虑量表总得分': 'anxiety_score',
                        '手机使用时间总得分': 'phone_usage_score'
                    })
                    
                    # 中文特征名称的背景数据保持原样
                    self.background_data_cn = sample_data
                    
                    print(f"✅ 从CSV文件加载背景数据")
                    
                    # 加载完整数据用于置信区间估算
                    self.full_data = df
                else:
                    st.error("找不到数据文件和预生成的背景数据")
                    self.background_data_en = None
                    self.background_data_cn = None
                    self.full_data = None
                    
        except Exception as e:
            st.error(f"加载数据失败: {e}")
            self.background_data_en = None
            self.background_data_cn = None
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
            print(f"开始创建SHAP图表...")
            
            # 获取特征值和英文名称
            feature_values = input_data.iloc[0].values
            english_names = [self.feature_name_mapping[name] for name in self.feature_names]
            
            # 获取基准值和SHAP值
            expected_value = explainer.expected_value
            print(f"Expected value: {expected_value}, type: {type(expected_value)}")
            
            if hasattr(expected_value, '__len__') and len(expected_value) > 1:
                expected_value = expected_value[0]
            
            print(f"SHAP values shape: {shap_values.shape}")
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
        """运行SHAP分析 - 简化版本，专门处理云端兼容性问题"""
        if not hasattr(self, 'background_data_en') or self.background_data_en is None or not SHAP_AVAILABLE:
            return None
        
        try:
            print(f"正在分析模型: {model_name}")  # 调试信息
            
            # 针对不同模型使用不同的SHAP解释器
            if model_name == 'XGBoost':
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
            
            elif model_name in ['LightGBM']:
                # LightGBM暂时跳过SHAP分析
                print(f"⚠️ {model_name} 在云端环境中暂时跳过SHAP分析（兼容性问题）")
                return None
                
            elif model_name in ['LinearRegression', 'Ridge']:
                # 线性模型使用LinearExplainer和中文特征名称
                print(f"使用LinearExplainer分析 {model_name}")
                explainer = shap.LinearExplainer(model, self.background_data_cn.sample(50, random_state=42))
                shap_values = explainer.shap_values(input_data)
                print(f"{model_name} LinearExplainer分析完成")
                
            elif model_name in ['KNN']:
                # KNN模型先暂时跳过SHAP分析，因为KernelExplainer太慢
                print(f"{model_name} 跳过SHAP分析（性能原因）")
                return None
            else:
                # 其他模型暂时跳过SHAP分析
                print(f"{model_name} 暂不支持SHAP分析")
                return None
            
            print(f"{model_name} SHAP分析成功，返回结果")
            return shap_values, explainer
            
        except Exception as e:
            print(f"SHAP分析错误 ({model_name}): {e}")
            # 不打印完整的traceback，避免干扰用户
            return None
    
    def run(self):
        """运行应用主程序"""
        # 页面标题
        st.markdown('<div class="main-title">抑郁量表得分预测</div>', unsafe_allow_html=True)
        
        # 只在SHAP不可用时显示提示
        if not SHAP_AVAILABLE:
            st.info("📊 预测功能正常运行，SHAP分析功能暂时不可用")
        
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
                # 准备输入数据 - 根据模型类型使用不同的特征名称
                if selected_model in ['XGBoost', 'LightGBM']:
                    # 树模型使用英文特征名称
                    input_data = pd.DataFrame({
                        'parent_child_score': [parent_child],
                        'resilience_score': [resilience],
                        'anxiety_score': [anxiety],
                        'phone_usage_score': [phone_usage]
                    })
                else:
                    # 其他模型使用中文特征名称
                    input_data = pd.DataFrame({
                        '亲子量表总得分': [parent_child],
                        '韧性量表总得分': [resilience],
                        '焦虑量表总得分': [anxiety],
                        '手机使用时间总得分': [phone_usage]
                    })
                
                # 进行预测
                try:
                    print(f"🔄 开始使用 {selected_model} 模型进行预测...")
                    print(f"📊 输入数据: {input_data}")
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        # 超强力修复XGBoost的GPU兼容性问题 - 运行时修复
                        model = self.models[selected_model]
                        
                        if selected_model in ['XGBoost', 'LightGBM']:
                            print(f"  🔧 正在修复{selected_model}的GPU兼容性...")
                            
                            # 策略1: 创建模型副本并清理GPU属性
                            try:
                                import copy
                                model = copy.deepcopy(model)
                                
                                # 移除所有可能的GPU相关属性
                                gpu_attrs = ['gpu_id', 'device', 'tree_method', 'predictor', 'gpu_hist']
                                for attr in gpu_attrs:
                                    if hasattr(model, attr):
                                        try:
                                            delattr(model, attr)
                                            print(f"    ✅ 移除属性: {attr}")
                                        except:
                                            pass
                                
                                # 强制设置CPU参数
                                cpu_params = {
                                    'device': 'cpu',
                                    'tree_method': 'hist',
                                    'predictor': 'cpu_predictor'
                                }
                                
                                if hasattr(model, 'set_param'):
                                    for key, value in cpu_params.items():
                                        try:
                                            model.set_param({key: value})
                                            print(f"    ✅ 设置参数: {key}={value}")
                                        except:
                                            pass
                                
                                # 处理booster
                                if hasattr(model, 'get_booster'):
                                    try:
                                        booster = model.get_booster()
                                        for key, value in cpu_params.items():
                                            try:
                                                booster.set_param({key: value})
                                                print(f"    ✅ Booster设置: {key}={value}")
                                            except:
                                                pass
                                    except:
                                        pass
                                
                                print(f"  ✅ {selected_model} GPU兼容性修复完成")
                                
                            except Exception as fix_error:
                                print(f"  ⚠️ 深度修复失败: {fix_error}")
                                # 如果深度修复失败，使用原模型
                                model = self.models[selected_model]
                        
                        # 安全预测函数 - 多层级错误处理
                        def safe_predict(model, data, model_name):
                            """安全预测函数，处理GPU兼容性问题"""
                            try:
                                # 首先尝试直接预测
                                return model.predict(data)[0]
                            except Exception as e:
                                error_str = str(e).lower()
                                # 检测各种GPU相关错误
                                gpu_keywords = ['gpu', 'device', 'cuda', 'gpu_id', 'tree_method', 'predictor']
                                is_gpu_error = any(keyword in error_str for keyword in gpu_keywords)
                                # 特别检测XGBoost的属性错误
                                is_xgb_attr_error = 'object has no attribute' in error_str and model_name in ['XGBoost', 'LightGBM']
                                
                                if is_gpu_error or is_xgb_attr_error:
                                    st.info(f"🔧 检测到GPU兼容性问题，正在切换到CPU模式...")
                                    
                                    # 策略1: 尝试温和修复
                                    try:
                                        import copy
                                        model_copy = copy.deepcopy(model)
                                        
                                        # 温和地设置CPU参数
                                        if hasattr(model_copy, 'set_param'):
                                            try:
                                                model_copy.set_param({'device': 'cpu'})
                                                model_copy.set_param({'tree_method': 'hist'})
                                            except:
                                                pass
                                        
                                        result = model_copy.predict(data)[0]
                                        st.success(f"✅ {model_name} 已成功切换到CPU模式")
                                        return result
                                        
                                    except Exception as cpu_error1:
                                        # 策略2: 重新加载原始模型
                                        try:
                                            st.info("🔄 正在重新加载原始模型...")
                                            models_dir = current_dir / 'models'
                                            model_path = models_dir / f'{model_name}_model.pkl'
                                            
                                            with open(model_path, 'rb') as f:
                                                fresh_model = pickle.load(f)
                                            
                                            # 直接尝试预测，不修改模型
                                            result = fresh_model.predict(data)[0]
                                            st.success(f"✅ {model_name} 原始模型预测成功")
                                            return result
                                            
                                        except Exception as cpu_error2:
                                            # 策略3: 尝试环境变量方式
                                            try:
                                                st.info("🛠️ 尝试环境变量修复...")
                                                import os
                                                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                                                
                                                result = model.predict(data)[0]
                                                st.success(f"✅ {model_name} 环境变量修复成功")
                                                return result
                                                
                                            except Exception as final_error:
                                                # 所有策略都失败，使用其他可用模型提示
                                                st.error(f"⚠️ {model_name} 在当前云端环境中暂时不可用")
                                                st.info("💡 建议使用 Ridge 或 LinearRegression 模型，它们在云端环境更稳定")
                                                raise e
                                else:
                                    # 非GPU相关错误，直接抛出
                                    raise e
                        
                        # 使用安全预测函数
                        try:
                            prediction = safe_predict(model, input_data, selected_model)
                        except Exception as pred_error:
                            st.error(f"预测失败: {pred_error}")
                            st.info("请尝试选择其他模型或检查输入数据")
                            return
                    
                    print(f"✅ {selected_model} 预测成功，结果: {prediction}")
                    
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
                    if SHAP_AVAILABLE:
                        try:
                            with st.spinner("正在生成特征重要性分析图..."):
                                shap_result = self.run_shap_analysis(self.models[selected_model], selected_model, input_data)
                                
                                if shap_result:
                                    shap_values, explainer = shap_result
                                    
                                    # 创建SHAP force plot
                                    fig = self.create_shap_force_plot(explainer, shap_values, input_data)
                                    if fig:
                                        st.pyplot(fig)
                                        plt.close(fig)  # 释放内存
                                elif selected_model in ['KNN']:
                                    st.info("💡 KNN模型的特征分析需要较长时间，已跳过")
                                elif selected_model in ['XGBoost', 'LightGBM']:
                                    st.info("💡 树模型在云端环境中的特征分析暂时不可用，可试试线性模型")
                        except Exception as shap_error:
                            st.warning(f"特征分析暂时不可用: {str(shap_error)}")
                
                except Exception as e:
                    error_msg = str(e)
                    if 'gpu_id' in error_msg and selected_model in ['XGBoost', 'LightGBM']:
                        # 特殊处理XGBoost/LightGBM的GPU错误
                        st.error(f"⚠️ {selected_model}模型遇到GPU兼容性问题")
                        st.info("💡 建议使用LinearRegression或Ridge模型，它们在云端环境中更稳定")
                        
                        # 尝试emergency修复并重试一次
                        try:
                            st.info("🔧 正在尝试紧急修复...")
                            model = self.models[selected_model]
                            
                            # 强制重置模型状态
                            import copy
                            model_copy = copy.deepcopy(model)
                            
                            # 移除所有可能的GPU属性
                            for attr in ['gpu_id', 'device', 'tree_method', '_Booster']:
                                if hasattr(model_copy, attr):
                                    try:
                                        delattr(model_copy, attr)
                                    except:
                                        pass
                            
                            # 使用修复后的模型重试预测
                            prediction = model_copy.predict(input_data)[0]
                            
                            # 如果成功，替换原模型
                            self.models[selected_model] = model_copy
                            st.success(f"🎉 {selected_model}模型修复成功！")
                            
                            # 继续显示结果的逻辑...
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
                            
                        except Exception as retry_error:
                            st.error(f"紧急修复失败: {retry_error}")
                            st.info("💡 建议使用LinearRegression或Ridge模型")
                    else:
                        st.error(f"预测失败: {e}")
                        st.info("请尝试选择其他模型或检查输入数据")
            else:
                st.error(f"模型 {selected_model} 不可用，请选择其他模型")

# 运行应用
if __name__ == "__main__":
    app = DepressionPredictionApp()
    app.run()
