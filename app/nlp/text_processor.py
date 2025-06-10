"""
文本预处理模块
负责清洗、标准化和预处理简历和职位描述文本
"""

import re
import string
import nltk
import spacy
from typing import List, Set, Dict, Tuple
from collections import Counter

# 确保NLTK数据已下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


class TextProcessor:
    """文本处理器类"""
    
    def __init__(self, language='en'):
        self.language = language
        self.stop_words = set(stopwords.words('english'))
        
        # 尝试加载spacy模型
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            print("Warning: spaCy model 'en_core_web_md' not found. Some features may be limited.")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本：去除特殊字符、多余空格等
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not text:
            return ""
        
        # 转换为小写
        text = text.lower()
        
        # 去除邮箱地址
        text = re.sub(r'\S+@\S+', '', text)
        
        # 去除URL
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # 去除电话号码
        text = re.sub(r'\+?\d{1,3}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}', '', text)
        
        # 保留字母、数字和基本标点
        text = re.sub(r'[^\w\s\-\+\#\.]', ' ', text)
        
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text)
        
        # 去除首尾空格
        text = text.strip()
        
        return text
    
    def extract_keywords(self, text: str, min_length: int = 2) -> List[str]:
        """
        提取关键词（去除停用词）
        
        Args:
            text: 输入文本
            min_length: 最小词长度
            
        Returns:
            关键词列表
        """
        if not text:
            return []
        
        # 分词
        tokens = word_tokenize(text.lower())
        
        # 过滤词汇
        keywords = []
        for token in tokens:
            # 去除停用词、标点符号、过短的词
            if (token not in self.stop_words and 
                token not in string.punctuation and 
                len(token) >= min_length and
                not token.isdigit()):
                keywords.append(token)
        
        return keywords
    
    def extract_technical_skills(self, text: str) -> Set[str]:
        """
        提取技术技能关键词
        
        Args:
            text: 输入文本
            
        Returns:
            技术技能集合
        """
        # 常见技术技能词汇
        tech_skills = {
            # 编程语言
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 
            'go', 'rust', 'kotlin', 'swift', 'scala', 'r', 'matlab', 'sql',
            
            # 框架和库
            'django', 'flask', 'fastapi', 'react', 'angular', 'vue', 'node.js', 'nodejs',
            'express', 'spring', 'laravel', 'rails', 'tensorflow', 'pytorch', 'pandas',
            'numpy', 'scikit-learn', 'keras', 'opencv',
            
            # 数据库
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
            'oracle', 'sqlite', 'dynamodb',
            
            # 云平台和工具
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github',
            'gitlab', 'linux', 'ubuntu', 'centos', 'nginx', 'apache',
            
            # 方法论
            'agile', 'scrum', 'devops', 'ci/cd', 'tdd', 'bdd', 'microservices',
            'rest', 'api', 'graphql', 'soap'
        }
        
        text_lower = text.lower()
        found_skills = set()
        
        # 查找完全匹配的技能
        for skill in tech_skills:
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                found_skills.add(skill)
        
        # 查找常见的技能变体
        skill_patterns = {
            'javascript': ['js', 'javascript'],
            'typescript': ['ts', 'typescript'],
            'node.js': ['nodejs', 'node.js', 'node js'],
            'c++': ['cpp', 'c++', 'c plus plus'],
            'c#': ['csharp', 'c#', 'c sharp'],
            'postgresql': ['postgres', 'postgresql'],
            'mongodb': ['mongo', 'mongodb'],
        }
        
        for canonical, patterns in skill_patterns.items():
            for pattern in patterns:
                if re.search(r'\b' + re.escape(pattern.lower()) + r'\b', text_lower):
                    found_skills.add(canonical)
                    break
        
        return found_skills
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        使用spaCy提取命名实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体字典 {实体类型: [实体列表]}
        """
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            entity_type = ent.label_
            entity_text = ent.text.strip()
            
            if entity_type not in entities:
                entities[entity_type] = []
            
            if entity_text not in entities[entity_type]:
                entities[entity_type].append(entity_text)
        
        return entities
    
    def get_sentence_embeddings(self, text: str) -> List[float]:
        """
        获取文本的词向量表示
        
        Args:
            text: 输入文本
            
        Returns:
            词向量列表
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        return doc.vector.tolist()
    
    def preprocess_for_matching(self, text: str) -> Dict:
        """
        为匹配分析预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            包含各种处理结果的字典
        """
        cleaned_text = self.clean_text(text)
        keywords = self.extract_keywords(cleaned_text)
        tech_skills = self.extract_technical_skills(text)
        entities = self.extract_entities(text)
        
        result = {
            'original': text,
            'cleaned': cleaned_text,
            'keywords': keywords,
            'tech_skills': list(tech_skills),
            'entities': entities,
            'keyword_freq': Counter(keywords)
        }
        
        # 如果spaCy可用，添加词向量
        if self.nlp:
            result['embeddings'] = self.get_sentence_embeddings(cleaned_text)
        
        return result 