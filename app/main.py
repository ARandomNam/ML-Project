"""
Resume Match Score Predictor - Streamlit主应用
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.nlp.text_processor import TextProcessor
from app.nlp.similarity import SimilarityCalculator
from app.utils.file_handler import FileHandler


def init_session_state():
    """初始化会话状态"""
    if 'processor' not in st.session_state:
        st.session_state.processor = TextProcessor()
    if 'calculator' not in st.session_state:
        st.session_state.calculator = SimilarityCalculator()
    if 'file_handler' not in st.session_state:
        st.session_state.file_handler = FileHandler()


def create_score_gauge(score):
    """创建分数仪表盘"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "匹配分数"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def create_skills_chart(skill_analysis):
    """创建技能分析图表"""
    matched_count = len(skill_analysis['matched_skills'])
    missing_count = len(skill_analysis['missing_skills'])
    extra_count = len(skill_analysis['extra_skills'])
    
    labels = ['匹配技能', '缺失技能', '额外技能']
    values = [matched_count, missing_count, extra_count]
    colors = ['#00cc96', '#ef553b', '#636efa']
    
    fig = px.pie(
        values=values, 
        names=labels, 
        title="技能分布",
        color_discrete_sequence=colors
    )
    
    fig.update_layout(height=400)
    return fig


def display_keywords_analysis(keyword_analysis):
    """展示关键词分析结果"""
    st.subheader("🔍 关键词分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "关键词匹配率", 
            f"{keyword_analysis['overlap_ratio']*100:.1f}%",
            f"{keyword_analysis['matched_count']}/{keyword_analysis['total_job_keywords']}"
        )
    
    with col2:
        st.metric(
            "匹配关键词数量", 
            keyword_analysis['matched_count'],
            f"总共 {keyword_analysis['total_job_keywords']} 个关键词"
        )
    
    if keyword_analysis['matched_keywords']:
        st.success("✅ **匹配的关键词:**")
        matched_text = ", ".join(keyword_analysis['matched_keywords'][:20])
        if len(keyword_analysis['matched_keywords']) > 20:
            matched_text += f" ... (还有 {len(keyword_analysis['matched_keywords'])-20} 个)"
        st.write(matched_text)
    
    if keyword_analysis['missing_keywords']:
        st.warning("⚠️ **缺失的关键词:**")
        missing_text = ", ".join(keyword_analysis['missing_keywords'][:20])
        if len(keyword_analysis['missing_keywords']) > 20:
            missing_text += f" ... (还有 {len(keyword_analysis['missing_keywords'])-20} 个)"
        st.write(missing_text)


def display_skills_analysis(skill_analysis):
    """展示技能分析结果"""
    st.subheader("💼 技能分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "技能匹配率", 
            f"{skill_analysis['skill_match_ratio']*100:.1f}%",
            f"{skill_analysis['matched_skills_count']}/{skill_analysis['total_required_skills']}"
        )
    
    with col2:
        # 创建技能分布图
        if any([skill_analysis['matched_skills'], skill_analysis['missing_skills'], skill_analysis['extra_skills']]):
            fig = create_skills_chart(skill_analysis)
            st.plotly_chart(fig, use_container_width=True)
    
    # 技能详情
    if skill_analysis['matched_skills']:
        st.success("✅ **匹配的技能:**")
        st.write(", ".join(skill_analysis['matched_skills']))
    
    if skill_analysis['missing_skills']:
        st.error("❌ **缺失的技能:**")
        st.write(", ".join(skill_analysis['missing_skills']))
    
    if skill_analysis['extra_skills']:
        st.info("➕ **额外的技能:**")
        st.write(", ".join(skill_analysis['extra_skills']))


def create_scores_dataframe(analysis_result, weights):
    """创建分数数据框"""
    scores_data = [
        {
            "指标": "TF-IDF相似度",
            "分数": f"{analysis_result['scores']['tfidf_similarity']:.3f}",
            "权重": f"{weights.get('tfidf_similarity', 0):.2f}"
        },
        {
            "指标": "spaCy语义相似度",
            "分数": f"{analysis_result['scores']['spacy_similarity'] or 0:.3f}",
            "权重": f"{weights.get('spacy_similarity', 0):.2f}"
        },
        {
            "指标": "关键词重叠度",
            "分数": f"{analysis_result['scores']['keyword_overlap']:.3f}",
            "权重": f"{weights.get('keyword_overlap', 0):.2f}"
        },
        {
            "指标": "技能匹配度",
            "分数": f"{analysis_result['scores']['skill_match']:.3f}",
            "权重": f"{weights.get('skill_match', 0):.2f}"
        }
    ]
    
    # 添加PyTorch深度学习分数
    if analysis_result.get('deep_learning_available') and 'pytorch_similarity' in analysis_result['scores']:
        scores_data.append({
            "指标": "🚀 PyTorch综合",
            "分数": f"{analysis_result['scores']['pytorch_similarity']:.3f}",
            "权重": f"{weights.get('pytorch_similarity', 0):.2f}"
        })
        
        # 添加各个模型的详细分数
        if 'sentence_bert' in analysis_result['scores']:
            scores_data.append({
                "指标": "📊 Sentence-BERT",
                "分数": f"{analysis_result['scores']['sentence_bert']:.3f}",
                "权重": "子模型"
            })
        
        if 'sentence_bert_large' in analysis_result['scores']:
            scores_data.append({
                "指标": "🎯 BERT-Large",
                "分数": f"{analysis_result['scores']['sentence_bert_large']:.3f}",
                "权重": "子模型"
            })
            
        if 'bert_base' in analysis_result['scores']:
            scores_data.append({
                "指标": "🔧 BERT-Base",
                "分数": f"{analysis_result['scores']['bert_base']:.3f}",
                "权重": "子模型"
            })
    
    return pd.DataFrame(scores_data)


def main():
    st.set_page_config(
        page_title="Resume Match Score Predictor",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 初始化会话状态
    init_session_state()
    
    # 标题和描述
    st.title("🎯 Resume Match Score Predictor")
    st.markdown("---")
    st.markdown(
        """
        这是一个基于NLP的智能简历匹配工具，能够分析简历与职位描述的匹配度，
        并提供详细的关键词和技能分析报告。
        """
    )
    
    # 侧边栏设置
    with st.sidebar:
        st.header("⚙️ 设置")
        
        # 分析方法选择
        st.subheader("分析方法")
        use_tfidf = st.checkbox("TF-IDF 相似度", value=True)
        use_spacy = st.checkbox("spaCy 语义相似度", value=st.session_state.calculator.spacy_available)
        use_keywords = st.checkbox("关键词匹配", value=True)
        use_skills = st.checkbox("技能匹配", value=True)
        
        # 深度学习选项
        deep_learning_available = getattr(st.session_state.calculator, 'deep_learning_available', False)
        use_sentence_bert = st.checkbox(
            "🚀 PyTorch 深度学习", 
            value=deep_learning_available,
            help="使用PyTorch和BERT模型进行语义相似度计算" if deep_learning_available else "需要安装PyTorch深度学习依赖"
        )
        
        if deep_learning_available:
            model_info = getattr(st.session_state.calculator.dl_calculator, 'get_model_info', lambda: {})()
            if model_info.get('available'):
                st.success(f"🎯 深度学习已启用 | 设备: {model_info.get('device', 'CPU')}")
                if model_info.get('cuda_available'):
                    st.info("⚡ GPU 加速可用")
        else:
            st.warning("⚠️ PyTorch深度学习功能未启用，请运行: pip install torch sentence-transformers transformers")
        
        # 权重调整
        st.subheader("权重设置")
        if use_tfidf:
            tfidf_weight = st.slider("TF-IDF权重", 0.0, 1.0, 0.3, 0.05)
        else:
            tfidf_weight = 0.0
            
        if use_spacy and st.session_state.calculator.spacy_available:
            spacy_weight = st.slider("spaCy权重", 0.0, 1.0, 0.2, 0.05)
        else:
            spacy_weight = 0.0
            
        if use_keywords:
            keyword_weight = st.slider("关键词权重", 0.0, 1.0, 0.25, 0.05)
        else:
            keyword_weight = 0.0
            
        if use_skills:
            skill_weight = st.slider("技能权重", 0.0, 1.0, 0.25, 0.05)
        else:
            skill_weight = 0.0
            
        if use_sentence_bert and deep_learning_available:
            pytorch_weight = st.slider("PyTorch深度学习权重", 0.0, 1.0, 0.25, 0.05)
        else:
            pytorch_weight = 0.0
        
        # 标准化权重
        total_weight = tfidf_weight + spacy_weight + keyword_weight + skill_weight + pytorch_weight
        if total_weight > 0:
            weights = {
                'tfidf_similarity': tfidf_weight / total_weight,
                'spacy_similarity': spacy_weight / total_weight,
                'keyword_overlap': keyword_weight / total_weight,
                'skill_match': skill_weight / total_weight,
                'pytorch_similarity': pytorch_weight / total_weight
            }
        else:
            weights = None
            st.error("请至少选择一种分析方法！")
    
    # 主界面
    tab1, tab2, tab3 = st.tabs(["📄 文本分析", "📁 文件上传", "📊 示例分析"])
    
    with tab1:
        st.header("文本输入分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("简历内容")
            resume_text = st.text_area(
                "请输入简历内容:",
                height=300,
                placeholder="请粘贴简历文本内容..."
            )
        
        with col2:
            st.subheader("职位描述")
            job_text = st.text_area(
                "请输入职位描述:",
                height=300,
                placeholder="请粘贴职位描述内容..."
            )
        
        if st.button("开始分析", type="primary", use_container_width=True):
            if resume_text and job_text and weights:
                with st.spinner("正在分析，请稍候..."):
                    # 预处理文本
                    resume_data = st.session_state.processor.preprocess_for_matching(resume_text)
                    job_data = st.session_state.processor.preprocess_for_matching(job_text)
                    
                    # 计算相似度
                    analysis_result = st.session_state.calculator.calculate_comprehensive_score(
                        resume_data, job_data, weights
                    )
                    
                    # 显示结果
                    st.success("分析完成！")
                    
                    # 总分显示
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.plotly_chart(
                            create_score_gauge(analysis_result['percentage_score']),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.subheader("📈 详细分数")
                        scores_df = create_scores_dataframe(analysis_result, weights)
                        st.dataframe(scores_df, use_container_width=True)
                        
                        # 显示深度学习模型信息
                        if analysis_result.get('deep_learning_available'):
                            model_info = analysis_result.get('model_info', {})
                            if model_info.get('available'):
                                st.caption(f"💡 深度学习: {model_info.get('device', 'CPU')} | 模型数量: {len(model_info.get('models', []))}")
                    
                    # 匹配解释
                    st.subheader("📝 匹配分析报告")
                    explanation = st.session_state.calculator.get_match_explanation(analysis_result)
                    st.info(explanation)
                    
                    # 详细分析
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        display_keywords_analysis(analysis_result['keyword_analysis'])
                    
                    with col2:
                        display_skills_analysis(analysis_result['skill_analysis'])
            
            else:
                st.error("请输入简历和职位描述内容，并选择至少一种分析方法！")
    
    with tab2:
        st.header("文件上传分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("上传简历文件")
            resume_file = st.file_uploader(
                "选择简历文件",
                type=['pdf', 'docx', 'txt'],
                help="支持PDF、DOCX、TXT格式"
            )
            
            resume_text_from_file = ""
            if resume_file:
                with st.spinner("正在处理简历文件..."):
                    result = st.session_state.file_handler.process_uploaded_file(
                        resume_file.read(), resume_file.name
                    )
                    
                    if result['success']:
                        resume_text_from_file = result['text']
                        st.success(f"✅ 成功处理文件: {result['filename']}")
                        with st.expander("查看提取的文本"):
                            st.text_area("提取的简历内容", resume_text_from_file, height=200)
                    else:
                        st.error(f"❌ 处理失败: {result['error']}")
        
        with col2:
            st.subheader("职位描述")
            job_text_for_file = st.text_area(
                "请输入职位描述:",
                height=300,
                placeholder="请粘贴职位描述内容...",
                key="job_text_file"
            )
        
        if st.button("分析上传文件", type="primary", use_container_width=True, key="analyze_file"):
            if resume_text_from_file and job_text_for_file and weights:
                with st.spinner("正在分析，请稍候..."):
                    # 预处理文本
                    resume_data = st.session_state.processor.preprocess_for_matching(resume_text_from_file)
                    job_data = st.session_state.processor.preprocess_for_matching(job_text_for_file)
                    
                    # 计算相似度
                    analysis_result = st.session_state.calculator.calculate_comprehensive_score(
                        resume_data, job_data, weights
                    )
                    
                    # 显示结果（同tab1的结果显示逻辑）
                    st.success("文件分析完成！")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.plotly_chart(
                            create_score_gauge(analysis_result['percentage_score']),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.subheader("📈 详细分数")
                        scores_df = create_scores_dataframe(analysis_result, weights)
                        st.dataframe(scores_df, use_container_width=True)
                        
                        # 显示深度学习模型信息
                        if analysis_result.get('deep_learning_available'):
                            model_info = analysis_result.get('model_info', {})
                            if model_info.get('available'):
                                st.caption(f"💡 深度学习: {model_info.get('device', 'CPU')} | 模型数量: {len(model_info.get('models', []))}")
                    
                    # 匹配解释
                    st.subheader("📝 匹配分析报告")
                    explanation = st.session_state.calculator.get_match_explanation(analysis_result)
                    st.info(explanation)
                    
                    # 详细分析
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        display_keywords_analysis(analysis_result['keyword_analysis'])
                    
                    with col2:
                        display_skills_analysis(analysis_result['skill_analysis'])
            
            else:
                st.error("请上传简历文件并输入职位描述！")
    
    with tab3:
        st.header("示例数据分析")
        st.markdown("使用项目内置的示例数据进行演示分析")
        
        if st.button("加载示例数据并分析", type="primary"):
            try:
                # 读取示例数据
                with open('data/sample/sample_resume.txt', 'r', encoding='utf-8') as f:
                    sample_resume = f.read()
                
                with open('data/sample/sample_job_description.txt', 'r', encoding='utf-8') as f:
                    sample_job = f.read()
                
                with st.spinner("正在分析示例数据..."):
                    # 预处理文本
                    resume_data = st.session_state.processor.preprocess_for_matching(sample_resume)
                    job_data = st.session_state.processor.preprocess_for_matching(sample_job)
                    
                    # 计算相似度
                    analysis_result = st.session_state.calculator.calculate_comprehensive_score(
                        resume_data, job_data, weights
                    )
                    
                    st.success("示例分析完成！")
                    
                    # 显示示例数据
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("示例简历")
                        st.text_area("", sample_resume, height=200, key="sample_resume_display")
                    
                    with col2:
                        st.subheader("示例职位描述")
                        st.text_area("", sample_job, height=200, key="sample_job_display")
                    
                    # 分析结果
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.plotly_chart(
                            create_score_gauge(analysis_result['percentage_score']),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.subheader("📈 详细分数")
                        scores_df = create_scores_dataframe(analysis_result, weights)
                        st.dataframe(scores_df, use_container_width=True)
                        
                        # 显示深度学习模型信息
                        if analysis_result.get('deep_learning_available'):
                            model_info = analysis_result.get('model_info', {})
                            if model_info.get('available'):
                                st.caption(f"💡 深度学习: {model_info.get('device', 'CPU')} | 模型数量: {len(model_info.get('models', []))}")
                    
                    # 匹配解释
                    st.subheader("📝 匹配分析报告")
                    explanation = st.session_state.calculator.get_match_explanation(analysis_result)
                    st.info(explanation)
                    
                    # 详细分析
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        display_keywords_analysis(analysis_result['keyword_analysis'])
                    
                    with col2:
                        display_skills_analysis(analysis_result['skill_analysis'])
            
            except FileNotFoundError:
                st.error("找不到示例数据文件，请确保data/sample/目录下有示例文件。")
            except Exception as e:
                st.error(f"分析示例数据时出错：{str(e)}")
    
    # 页脚
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🎯 Resume Match Score Predictor | 
            基于NLP的智能简历匹配工具 | 
            Powered by Streamlit</p>
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main() 