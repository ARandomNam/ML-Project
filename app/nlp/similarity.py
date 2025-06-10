"""
相似度计算模块
实现TF-IDF + 余弦相似度和spaCy词向量相似度计算
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import spacy


class SimilarityCalculator:
    """相似度计算器类"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        
        # 尝试加载spacy模型
        try:
            self.nlp = spacy.load("en_core_web_md")
            self.spacy_available = True
        except OSError:
            print("Warning: spaCy model 'en_core_web_md' not found. Using TF-IDF only.")
            self.nlp = None
            self.spacy_available = False
    
    def calculate_tfidf_similarity(self, resume_text: str, job_text: str) -> float:
        """
        使用TF-IDF + 余弦相似度计算文本相似度
        
        Args:
            resume_text: 简历文本
            job_text: 职位描述文本
            
        Returns:
            相似度分数 (0-1)
        """
        if not resume_text or not job_text:
            return 0.0
        
        # 创建TF-IDF向量
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),  # 包含单词和二元组
            lowercase=True,
            min_df=1,
            max_df=0.95
        )
        
        # 拟合并转换文本
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([resume_text, job_text])
            
            # 计算余弦相似度
            similarity_matrix = cosine_similarity(tfidf_matrix)
            similarity_score = similarity_matrix[0][1]
            
            return max(0.0, min(1.0, similarity_score))
        
        except Exception as e:
            print(f"Error calculating TF-IDF similarity: {e}")
            return 0.0
    
    def calculate_spacy_similarity(self, resume_text: str, job_text: str) -> float:
        """
        使用spaCy词向量计算语义相似度
        
        Args:
            resume_text: 简历文本
            job_text: 职位描述文本
            
        Returns:
            相似度分数 (0-1)
        """
        if not self.spacy_available or not resume_text or not job_text:
            return 0.0
        
        try:
            # 获取文档向量
            resume_doc = self.nlp(resume_text)
            job_doc = self.nlp(job_text)
            
            # 计算相似度
            similarity = resume_doc.similarity(job_doc)
            
            return max(0.0, min(1.0, similarity))
        
        except Exception as e:
            print(f"Error calculating spaCy similarity: {e}")
            return 0.0
    
    def calculate_keyword_overlap(self, resume_keywords: List[str], 
                                job_keywords: List[str]) -> Dict[str, Any]:
        """
        计算关键词重叠度
        
        Args:
            resume_keywords: 简历关键词列表
            job_keywords: 职位关键词列表
            
        Returns:
            包含重叠分析的字典
        """
        if not resume_keywords or not job_keywords:
            return {
                'overlap_ratio': 0.0,
                'matched_keywords': [],
                'missing_keywords': job_keywords,
                'unique_resume_keywords': resume_keywords
            }
        
        resume_set = set(resume_keywords)
        job_set = set(job_keywords)
        
        # 计算交集和差集
        matched = resume_set.intersection(job_set)
        missing = job_set - resume_set
        unique_resume = resume_set - job_set
        
        # 计算重叠比例
        overlap_ratio = len(matched) / len(job_set) if job_set else 0.0
        
        return {
            'overlap_ratio': overlap_ratio,
            'matched_keywords': list(matched),
            'missing_keywords': list(missing),
            'unique_resume_keywords': list(unique_resume),
            'total_job_keywords': len(job_set),
            'total_resume_keywords': len(resume_set),
            'matched_count': len(matched)
        }
    
    def calculate_skill_match(self, resume_skills: List[str], 
                            job_skills: List[str]) -> Dict[str, Any]:
        """
        计算技术技能匹配度
        
        Args:
            resume_skills: 简历技能列表
            job_skills: 职位技能列表
            
        Returns:
            技能匹配分析结果
        """
        if not job_skills:
            return {
                'skill_match_ratio': 1.0,
                'matched_skills': [],
                'missing_skills': [],
                'extra_skills': resume_skills or []
            }
        
        if not resume_skills:
            return {
                'skill_match_ratio': 0.0,
                'matched_skills': [],
                'missing_skills': job_skills,
                'extra_skills': []
            }
        
        resume_skills_set = set(skill.lower() for skill in resume_skills)
        job_skills_set = set(skill.lower() for skill in job_skills)
        
        # 查找匹配的技能
        matched_skills = resume_skills_set.intersection(job_skills_set)
        missing_skills = job_skills_set - resume_skills_set
        extra_skills = resume_skills_set - job_skills_set
        
        # 计算匹配比例
        skill_match_ratio = len(matched_skills) / len(job_skills_set)
        
        return {
            'skill_match_ratio': skill_match_ratio,
            'matched_skills': list(matched_skills),
            'missing_skills': list(missing_skills),
            'extra_skills': list(extra_skills),
            'total_required_skills': len(job_skills_set),
            'matched_skills_count': len(matched_skills)
        }
    
    def calculate_comprehensive_score(self, resume_data: Dict, 
                                    job_data: Dict, 
                                    weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        计算综合匹配分数
        
        Args:
            resume_data: 简历处理数据
            job_data: 职位处理数据
            weights: 各项指标权重
            
        Returns:
            综合分析结果
        """
        if weights is None:
            weights = {
                'tfidf_similarity': 0.3,
                'spacy_similarity': 0.2,
                'keyword_overlap': 0.25,
                'skill_match': 0.25
            }
        
        # 计算各项相似度指标
        tfidf_score = self.calculate_tfidf_similarity(
            resume_data['cleaned'], 
            job_data['cleaned']
        )
        
        spacy_score = self.calculate_spacy_similarity(
            resume_data['cleaned'], 
            job_data['cleaned']
        ) if self.spacy_available else 0.0
        
        keyword_analysis = self.calculate_keyword_overlap(
            resume_data['keywords'], 
            job_data['keywords']
        )
        
        skill_analysis = self.calculate_skill_match(
            resume_data['tech_skills'], 
            job_data['tech_skills']
        )
        
        # 如果spaCy不可用，重新分配权重
        if not self.spacy_available:
            weights['tfidf_similarity'] = 0.4
            weights['keyword_overlap'] = 0.3
            weights['skill_match'] = 0.3
            weights['spacy_similarity'] = 0.0
        
        # 计算加权综合分数
        comprehensive_score = (
            weights['tfidf_similarity'] * tfidf_score +
            weights['spacy_similarity'] * spacy_score +
            weights['keyword_overlap'] * keyword_analysis['overlap_ratio'] +
            weights['skill_match'] * skill_analysis['skill_match_ratio']
        )
        
        # 计算百分制分数
        percentage_score = comprehensive_score * 100
        
        return {
            'overall_score': comprehensive_score,
            'percentage_score': round(percentage_score, 1),
            'scores': {
                'tfidf_similarity': round(tfidf_score, 3),
                'spacy_similarity': round(spacy_score, 3) if self.spacy_available else None,
                'keyword_overlap': round(keyword_analysis['overlap_ratio'], 3),
                'skill_match': round(skill_analysis['skill_match_ratio'], 3)
            },
            'keyword_analysis': keyword_analysis,
            'skill_analysis': skill_analysis,
            'weights_used': weights,
            'spacy_available': self.spacy_available
        }
    
    def get_match_explanation(self, analysis_result: Dict) -> str:
        """
        生成匹配结果的文字解释
        
        Args:
            analysis_result: 分析结果
            
        Returns:
            解释文本
        """
        score = analysis_result['percentage_score']
        
        if score >= 80:
            level = "优秀匹配"
            description = "简历与职位要求高度匹配"
        elif score >= 60:
            level = "良好匹配"
            description = "简历基本符合职位要求"
        elif score >= 40:
            level = "中等匹配"
            description = "简历部分符合职位要求"
        elif score >= 20:
            level = "较低匹配"
            description = "简历与职位要求匹配度较低"
        else:
            level = "不匹配"
            description = "简历与职位要求不匹配"
        
        skill_analysis = analysis_result['skill_analysis']
        matched_skills = len(skill_analysis['matched_skills'])
        total_skills = skill_analysis['total_required_skills']
        
        explanation = f"""
        匹配等级：{level} ({score}%)
        
        {description}
        
        技能匹配：{matched_skills}/{total_skills} 项技能匹配
        匹配的技能：{', '.join(skill_analysis['matched_skills'][:5])}{'...' if len(skill_analysis['matched_skills']) > 5 else ''}
        缺失的技能：{', '.join(skill_analysis['missing_skills'][:5])}{'...' if len(skill_analysis['missing_skills']) > 5 else ''}
        """
        
        return explanation.strip() 