import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
from pathlib import Path

# 机器学习库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# 导入XGBoost和LightGBM（如果可用）
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# 获取当前目录路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

warnings.filterwarnings("ignore")

# 设置页面配置
st.set_page_config(
    page_title="抑郁量表得分预测应用",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-result {
        font-size: 2rem;
        color: #d62728;
        text-align: center;
        padding: 1rem;
        border: 2px solid #d62728;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    .model-info {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .feature-info {
        background-color: #f0f8ff;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

class DepressionPredictionApp:
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.background_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # 特征名称映射
        self.feature_names = ['亲子量表总得分', '韧性量表总得分', '焦虑量表总得分', '手机使用时间总得分']
        self.feature_name_mapping = {
            '亲子量表总得分': 'Parent-Child Relationship',
            '韧性量表总得分': 'Resilience', 
            '焦虑量表总得分': 'Anxiety',
            '手机使用时间总得分': 'Phone Usage Time'
        }
        
        # 特征范围
        self.feature_ranges = {
            '亲子量表总得分': (8, 50),
            '韧性量表总得分': (0, 40),
            '焦虑量表总得分': (0, 20),
            '手机使用时间总得分': (0, 60)
        }
        
        # 模型描述
        self.model_descriptions = {
            'XGBoost': '极端梯度提升，高性能树模型',
            'LightGBM': '轻量级梯度提升，快速高效',
            'RandomForest': '随机森林，稳定可靠',
            'GradientBoosting': '梯度提升，经典集成方法',
            'ExtraTrees': '极端随机树，降低过拟合',
            'SVM': '支持向量机，非线性映射',
            'ANN': '人工神经网络，深度学习',
            'KNN': 'K近邻算法，基于相似性',
            'DecisionTree': '决策树，可解释性强',
            'AdaBoost': '自适应提升，错误加权',
            'LinearRegression': '线性回归，基础模型',
            'Ridge': '岭回归，正则化线性模型'
        }
        
        # 初始化
        self.load_data()
        self.load_models()
        
    def load_data(self):
        """加载数据"""
        try:
            data_path = current_dir / 'data' / '量表总分完整数据.csv'
            if data_path.exists():
                df = pd.read_csv(data_path)
                
                # 准备特征和目标变量
                X = df[self.feature_names]
                y = df['抑郁量表总得分']
                
                # 分割数据
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # 随机采样1000个样本作为背景数据
                self.background_data = X.sample(n=min(1000, len(X)), random_state=42)
                
                st.success(f"✅ 数据加载成功！训练集: {len(self.X_train)} 样本，测试集: {len(self.X_test)} 样本")
            else:
                st.error("❌ 找不到数据文件")
                return False
        except Exception as e:
            st.error(f"❌ 加载数据失败: {e}")
            return False
        return True
    
    def load_models(self):
        """加载训练好的模型"""
        models_dir = current_dir / 'models'
        
        # 模型文件名映射
        model_files = {
            'XGBoost': 'XGBoost_model.pkl',
            'LightGBM': 'LightGBM_model.pkl',
            'RandomForest': 'RandomForest_model.pkl',
            'GradientBoosting': 'GradientBoosting_model.pkl',
            'ExtraTrees': 'ExtraTrees_model.pkl',
            'SVM': 'SVM_model.pkl',
            'ANN': 'ANN_model.pkl',
            'KNN': 'KNN_model.pkl',
            'DecisionTree': 'DecisionTree_model.pkl',
            'AdaBoost': 'AdaBoost_model.pkl',
            'LinearRegression': 'LinearRegression_model.pkl',
            'Ridge': 'Ridge_model.pkl'
        }
        
        loaded_count = 0
        failed_count = 0
        
        for model_name, file_name in model_files.items():
            model_path = models_dir / file_name
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    loaded_count += 1
                except Exception as e:
                    st.warning(f"⚠️ 无法加载模型 {model_name}: {e}")
                    failed_count += 1
        
        if loaded_count > 0:
            st.info(f"📊 成功加载 {loaded_count} 个预训练模型，{failed_count} 个模型加载失败")
        else:
            st.warning("⚠️ 没有可用的预训练模型，将使用实时训练功能")
    
    def train_fresh_models(self, selected_models):
        """训练新的模型"""
        if self.X_train is None or self.y_train is None:
            st.error("❌ 训练数据不可用")
            return {}
        
        trained_models = {}
        
        # 定义模型
        model_definitions = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'DecisionTree': DecisionTreeRegressor(random_state=42),
            'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'SVM': SVR(kernel='rbf', C=1.0),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'ANN': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        # 添加XGBoost（如果可用）
        if XGBOOST_AVAILABLE:
            model_definitions['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
        
        # 添加LightGBM（如果可用）
        if LIGHTGBM_AVAILABLE:
            model_definitions['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_name in enumerate(selected_models):
            if model_name in model_definitions:
                try:
                    status_text.text(f"正在训练 {model_name} 模型...")
                    
                    model = model_definitions[model_name]
                    model.fit(self.X_train, self.y_train)
                    
                    # 评估模型
                    y_pred = model.predict(self.X_test)
                    mse = mean_squared_error(self.y_test, y_pred)
                    r2 = r2_score(self.y_test, y_pred)
                    
                    trained_models[model_name] = {
                        'model': model,
                        'mse': mse,
                        'r2': r2
                    }
                    
                    progress_bar.progress((i + 1) / len(selected_models))
                    
                except Exception as e:
                    st.warning(f"⚠️ 训练模型 {model_name} 失败: {e}")
        
        status_text.text("✅ 模型训练完成！")
        progress_bar.empty()
        
        return trained_models
    
    def create_shap_waterfall_plot(self, shap_values, expected_value, feature_values, feature_names):
        """创建SHAP瀑布图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 准备数据
        values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        features = [f"{name} = {val:.1f}" for name, val in zip(feature_names, feature_values)]
        
        # 绘制基准线
        ax.axhline(y=expected_value, color='gray', linestyle='--', alpha=0.7, label=f'基准值: {expected_value:.2f}')
        
        # 绘制特征贡献
        positions = range(len(features))
        colors = ['red' if val < 0 else 'blue' for val in values]
        
        # 绘制条形图
        bars = ax.bar(positions, np.abs(values), color=colors, alpha=0.7)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{val:+.2f}', ha='center', va='bottom', fontsize=10)
        
        # 设置标签和标题
        ax.set_xticks(positions)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.set_ylabel('SHAP值 (对预测的影响)')
        ax.set_title('SHAP特征贡献分析 - 各特征对预测结果的影响')
        
        # 添加预测结果线
        final_prediction = expected_value + sum(values)
        ax.axhline(y=final_prediction, color='green', linestyle='-', linewidth=2, 
                  label=f'预测结果: {final_prediction:.2f}')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def run_shap_analysis(self, model, model_name, input_data):
        """运行SHAP分析"""
        if self.background_data is None:
            st.error("❌ 背景数据未加载，无法进行SHAP分析")
            return None
        
        try:
            # 初始化SHAP解释器
            if model_name in ['XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting', 
                            'ExtraTrees', 'DecisionTree', 'AdaBoost']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_data)
            else:
                # 对于其他模型使用KernelExplainer
                explainer = shap.KernelExplainer(model.predict, self.background_data.sample(100))
                shap_values = explainer.shap_values(input_data)
            
            expected_value = explainer.expected_value
            if hasattr(expected_value, '__len__') and len(expected_value) > 1:
                expected_value = expected_value[0]
            
            return shap_values, expected_value, explainer
            
        except Exception as e:
            st.error(f"❌ SHAP分析失败: {e}")
            return None
    
    def render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.header("🔧 模型选择")
        
        # 显示可用模型状态
        if self.models:
            st.sidebar.success(f"✅ {len(self.models)} 个预训练模型可用")
        else:
            st.sidebar.warning("⚠️ 无预训练模型，将实时训练")
        
        # 模型选择
        available_models = list(self.model_descriptions.keys())
        selected_models = st.sidebar.multiselect(
            "选择要使用的模型：",
            available_models,
            default=['XGBoost', 'LightGBM', 'RandomForest'] if not self.models else list(self.models.keys())[:3],
            help="选择一个或多个模型进行预测"
        )
        
        # 训练选项
        if not self.models or st.sidebar.checkbox("重新训练模型"):
            self.use_fresh_training = True
        else:
            self.use_fresh_training = False
        
        return selected_models
    
    def render_input_form(self):
        """渲染输入表单"""
        st.markdown('<div class="sub-header">📊 特征输入</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 亲子量表得分
            st.markdown('<div class="feature-info">亲子量表总得分 (范围: 8-50)</div>', 
                       unsafe_allow_html=True)
            parent_child = st.number_input(
                "亲子量表总得分",
                min_value=8, max_value=50, value=None,
                step=1, help="请输入8-50之间的整数"
            )
            
            # 焦虑量表得分
            st.markdown('<div class="feature-info">焦虑量表总得分 (范围: 0-20)</div>', 
                       unsafe_allow_html=True)
            anxiety = st.number_input(
                "焦虑量表总得分",
                min_value=0, max_value=20, value=None,
                step=1, help="请输入0-20之间的整数"
            )
        
        with col2:
            # 韧性量表得分
            st.markdown('<div class="feature-info">韧性量表总得分 (范围: 0-40)</div>', 
                       unsafe_allow_html=True)
            resilience = st.number_input(
                "韧性量表总得分",
                min_value=0, max_value=40, value=None,
                step=1, help="请输入0-40之间的整数"
            )
            
            # 手机使用时间得分
            st.markdown('<div class="feature-info">手机使用时间总得分 (范围: 0-60)</div>', 
                       unsafe_allow_html=True)
            phone_usage = st.number_input(
                "手机使用时间总得分",
                min_value=0, max_value=60, value=None,
                step=1, help="请输入0-60之间的整数"
            )
        
        return parent_child, resilience, anxiety, phone_usage
    
    def validate_inputs(self, parent_child, resilience, anxiety, phone_usage):
        """验证输入"""
        if any(val is None for val in [parent_child, resilience, anxiety, phone_usage]):
            return False, "请填写所有特征值"
        return True, ""
    
    def run_predictions(self, selected_models, input_data):
        """运行预测"""
        predictions = {}
        
        # 如果需要重新训练或没有预训练模型
        if self.use_fresh_training or not self.models:
            st.info("🔄 正在训练新模型...")
            self.trained_models = self.train_fresh_models(selected_models)
            
            for model_name in selected_models:
                if model_name in self.trained_models:
                    try:
                        model_info = self.trained_models[model_name]
                        pred = model_info['model'].predict(input_data)[0]
                        predictions[model_name] = pred
                    except Exception as e:
                        st.error(f"❌ 模型 {model_name} 预测失败: {e}")
        else:
            # 使用预训练模型
            for model_name in selected_models:
                if model_name in self.models:
                    try:
                        pred = self.models[model_name].predict(input_data)[0]
                        predictions[model_name] = pred
                    except Exception as e:
                        st.error(f"❌ 模型 {model_name} 预测失败: {e}")
        
        return predictions
    
    def render_results(self, predictions, input_data, selected_models):
        """渲染预测结果"""
        if not predictions:
            st.warning("❌ 没有可用的预测结果")
            return
        
        # 显示预测结果
        st.markdown('<div class="sub-header">🎯 预测结果</div>', unsafe_allow_html=True)
        
        # 计算平均预测
        avg_prediction = np.mean(list(predictions.values()))
        
        st.markdown(f'''
        <div class="prediction-result">
            平均预测的抑郁量表得分: {avg_prediction:.2f}
        </div>
        ''', unsafe_allow_html=True)
        
        # 显示各模型预测结果
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 各模型预测结果")
            results_df = pd.DataFrame({
                '模型': list(predictions.keys()),
                '预测得分': [f"{pred:.2f}" for pred in predictions.values()]
            })
            st.dataframe(results_df, use_container_width=True)
        
        with col2:
            st.subheader("📊 预测分布")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(range(len(predictions)), list(predictions.values()), 
                  color='skyblue', alpha=0.7)
            ax.axhline(y=avg_prediction, color='red', linestyle='--', 
                      label=f'平均值: {avg_prediction:.2f}')
            ax.set_xticks(range(len(predictions)))
            ax.set_xticklabels(list(predictions.keys()), rotation=45, ha='right')
            ax.set_ylabel('预测得分')
            ax.set_title('各模型预测结果对比')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        # SHAP分析
        self.render_shap_analysis(input_data, selected_models)
    
    def render_shap_analysis(self, input_data, selected_models):
        """渲染SHAP分析"""
        st.markdown('<div class="sub-header">🔍 SHAP解释性分析</div>', unsafe_allow_html=True)
        
        # 选择要分析的模型
        available_models = []
        if self.use_fresh_training or not self.models:
            available_models = [name for name in selected_models if name in self.trained_models]
        else:
            available_models = [name for name in selected_models if name in self.models]
        
        if not available_models:
            st.warning("❌ 没有可用于SHAP分析的模型")
            return
        
        shap_model = st.selectbox(
            "选择要进行SHAP分析的模型：",
            available_models,
            help="选择一个模型来查看其预测的详细解释"
        )
        
        if shap_model:
            # 获取模型
            if self.use_fresh_training or not self.models:
                model = self.trained_models[shap_model]['model']
            else:
                model = self.models[shap_model]
            
            with st.spinner(f"正在为{shap_model}模型生成SHAP分析..."):
                shap_result = self.run_shap_analysis(model, shap_model, input_data)
                
                if shap_result:
                    shap_values, expected_value, explainer = shap_result
                    
                    # 准备特征值和名称
                    feature_values = input_data.iloc[0].values
                    english_names = [self.feature_name_mapping[name] for name in self.feature_names]
                    
                    # 创建瀑布图
                    st.subheader("🌊 SHAP瀑布图 - 特征贡献分析")
                    
                    waterfall_fig = self.create_shap_waterfall_plot(
                        shap_values, expected_value, feature_values, english_names
                    )
                    st.pyplot(waterfall_fig)
                    
                    # 显示详细的SHAP值
                    st.subheader("📋 详细SHAP分析")
                    
                    # 准备SHAP数据
                    if len(shap_values.shape) > 1:
                        shap_vals = shap_values[0]
                    else:
                        shap_vals = shap_values
                    
                    shap_df = pd.DataFrame({
                        '特征': english_names,
                        '特征值': feature_values,
                        'SHAP值': shap_vals,
                        '贡献程度': ['正向' if val > 0 else '负向' for val in shap_vals],
                        '影响大小': np.abs(shap_vals)
                    }).sort_values('影响大小', ascending=False)
                    
                    st.dataframe(shap_df, use_container_width=True)
                    
                    # 解释说明
                    st.info("""
                    **SHAP分析解释：**
                    - **基准值**: 模型在训练数据上的平均预测值
                    - **SHAP值**: 每个特征对最终预测的贡献，正值表示增加预测值，负值表示减少预测值
                    - **预测结果**: 基准值加上所有特征的SHAP值之和
                    - **特征重要性**: 按照SHAP值的绝对值大小排序，值越大表示该特征对预测结果的影响越大
                    """)
    
    def run(self):
        """运行应用主程序"""
        # 页面标题
        st.markdown('<div class="main-header">🧠 抑郁量表得分预测应用</div>', 
                   unsafe_allow_html=True)
        
        # 应用说明
        st.markdown("""
        <div class="model-info">
        <h4>📋 应用说明</h4>
        <p>本应用使用多种机器学习模型来预测抑郁量表得分。可以使用预训练模型或实时训练新模型。</p>
        <p><strong>特征说明：</strong></p>
        <ul>
            <li><strong>亲子量表总得分</strong>: 反映亲子关系质量 (8-50分)</li>
            <li><strong>韧性量表总得分</strong>: 反映心理韧性水平 (0-40分)</li>
            <li><strong>焦虑量表总得分</strong>: 反映焦虑程度 (0-20分)</li>
            <li><strong>手机使用时间总得分</strong>: 反映手机使用情况 (0-60分)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # 侧边栏：模型选择
        selected_models = self.render_sidebar()
        
        if not selected_models:
            st.warning("⚠️ 请在左侧选择至少一个模型进行预测")
            return
        
        # 显示已选择的模型
        st.markdown(f"**已选择模型**: {', '.join(selected_models)}")
        
        # 输入表单
        parent_child, resilience, anxiety, phone_usage = self.render_input_form()
        
        # 预测按钮
        if st.button("🚀 开始预测", type="primary", use_container_width=True):
            # 验证输入
            is_valid, error_msg = self.validate_inputs(parent_child, resilience, anxiety, phone_usage)
            
            if not is_valid:
                st.error(error_msg)
                return
            
            # 准备输入数据
            input_data = pd.DataFrame({
                '亲子量表总得分': [parent_child],
                '韧性量表总得分': [resilience],
                '焦虑量表总得分': [anxiety],
                '手机使用时间总得分': [phone_usage]
            })
            
            # 运行预测
            with st.spinner("正在进行预测..."):
                predictions = self.run_predictions(selected_models, input_data)
            
            # 显示结果
            self.render_results(predictions, input_data, selected_models)

# 运行应用
if __name__ == "__main__":
    app = DepressionPredictionApp()
    app.run() 