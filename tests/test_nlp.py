"""
NLP模块测试
"""

import unittest
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.nlp.text_processor import TextProcessor
from app.nlp.similarity import SimilarityCalculator


class TestTextProcessor(unittest.TestCase):
    """文本处理器测试"""
    
    def setUp(self):
        self.processor = TextProcessor()
    
    def test_clean_text(self):
        """测试文本清洗"""
        text = "Hello World! Email: test@example.com Phone: +1-234-567-8900"
        cleaned = self.processor.clean_text(text)
        
        self.assertNotIn("@", cleaned)
        self.assertNotIn("+1-234-567-8900", cleaned)
        self.assertIn("hello world", cleaned)
    
    def test_extract_keywords(self):
        """测试关键词提取"""
        text = "Python developer with Django experience"
        keywords = self.processor.extract_keywords(text)
        
        self.assertIn("python", keywords)
        self.assertIn("developer", keywords)
        self.assertIn("django", keywords)
        self.assertNotIn("with", keywords)  # 停用词应被过滤
    
    def test_extract_technical_skills(self):
        """测试技术技能提取"""
        text = "Experience with Python, JavaScript, React, and PostgreSQL"
        skills = self.processor.extract_technical_skills(text)
        
        self.assertIn("python", skills)
        self.assertIn("javascript", skills)
        self.assertIn("react", skills)
        self.assertIn("postgresql", skills)
    
    def test_preprocess_for_matching(self):
        """测试完整预处理流程"""
        text = "Senior Python Developer with 5 years experience in Django and React"
        result = self.processor.preprocess_for_matching(text)
        
        self.assertIn("cleaned", result)
        self.assertIn("keywords", result)
        self.assertIn("tech_skills", result)
        self.assertIn("entities", result)
        self.assertIn("keyword_freq", result)
        
        self.assertTrue(len(result["keywords"]) > 0)
        self.assertTrue(len(result["tech_skills"]) > 0)


class TestSimilarityCalculator(unittest.TestCase):
    """相似度计算器测试"""
    
    def setUp(self):
        self.calculator = SimilarityCalculator()
    
    def test_tfidf_similarity(self):
        """测试TF-IDF相似度计算"""
        text1 = "Python developer with Django experience"
        text2 = "Looking for Python Django developer"
        
        similarity = self.calculator.calculate_tfidf_similarity(text1, text2)
        
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        self.assertGreater(similarity, 0.3)  # 应该有较高相似度
    
    def test_keyword_overlap(self):
        """测试关键词重叠计算"""
        resume_keywords = ["python", "django", "react", "javascript"]
        job_keywords = ["python", "django", "postgresql", "git"]
        
        result = self.calculator.calculate_keyword_overlap(resume_keywords, job_keywords)
        
        self.assertEqual(result["matched_count"], 2)  # python, django
        self.assertEqual(len(result["missing_keywords"]), 2)  # postgresql, git
        self.assertEqual(len(result["unique_resume_keywords"]), 2)  # react, javascript
        self.assertAlmostEqual(result["overlap_ratio"], 0.5)  # 2/4
    
    def test_skill_match(self):
        """测试技能匹配计算"""
        resume_skills = ["python", "django", "react"]
        job_skills = ["python", "django", "postgresql"]
        
        result = self.calculator.calculate_skill_match(resume_skills, job_skills)
        
        self.assertEqual(result["matched_skills_count"], 2)
        self.assertEqual(len(result["missing_skills"]), 1)
        self.assertAlmostEqual(result["skill_match_ratio"], 2/3)
    
    def test_comprehensive_score(self):
        """测试综合分数计算"""
        # 模拟预处理数据
        resume_data = {
            "cleaned": "python developer django experience",
            "keywords": ["python", "developer", "django", "experience"],
            "tech_skills": ["python", "django"]
        }
        
        job_data = {
            "cleaned": "looking python django developer",
            "keywords": ["looking", "python", "django", "developer"],
            "tech_skills": ["python", "django", "postgresql"]
        }
        
        result = self.calculator.calculate_comprehensive_score(resume_data, job_data)
        
        self.assertIn("overall_score", result)
        self.assertIn("percentage_score", result)
        self.assertIn("scores", result)
        self.assertIn("keyword_analysis", result)
        self.assertIn("skill_analysis", result)
        
        self.assertGreaterEqual(result["overall_score"], 0.0)
        self.assertLessEqual(result["overall_score"], 1.0)
        self.assertGreaterEqual(result["percentage_score"], 0.0)
        self.assertLessEqual(result["percentage_score"], 100.0)


if __name__ == "__main__":
    unittest.main() 