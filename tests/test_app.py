"""
应用模块测试
"""

import unittest
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.file_handler import FileHandler


class TestFileHandler(unittest.TestCase):
    """文件处理器测试"""
    
    def setUp(self):
        self.handler = FileHandler()
    
    def test_supported_formats(self):
        """测试支持的文件格式"""
        self.assertTrue(self.handler.is_supported_format("resume.pdf"))
        self.assertTrue(self.handler.is_supported_format("resume.docx"))
        self.assertTrue(self.handler.is_supported_format("resume.txt"))
        self.assertFalse(self.handler.is_supported_format("resume.jpg"))
    
    def test_extract_text_from_txt(self):
        """测试从TXT文件提取文本"""
        test_content = "This is a test resume content".encode('utf-8')
        result = self.handler.extract_text_from_txt(test_content)
        
        self.assertEqual(result, "This is a test resume content")
    
    def test_clean_extracted_text(self):
        """测试文本清理"""
        dirty_text = """
        
        Line 1
        
        
        Line 2   
        
        """
        
        cleaned = self.handler.clean_extracted_text(dirty_text)
        lines = cleaned.split('\n')
        
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], "Line 1")
        self.assertEqual(lines[1], "Line 2")
    
    def test_validate_file_size(self):
        """测试文件大小验证"""
        small_content = b"small file"
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        
        self.assertTrue(self.handler.validate_file_size(small_content))
        self.assertFalse(self.handler.validate_file_size(large_content))
    
    def test_process_uploaded_file_txt(self):
        """测试处理上传的TXT文件"""
        test_content = "Test resume content with Python skills".encode('utf-8')
        result = self.handler.process_uploaded_file(test_content, "resume.txt")
        
        self.assertTrue(result['success'])
        self.assertIn("Python", result['text'])
        self.assertEqual(result['filename'], "resume.txt")
        self.assertIsNone(result['error'])
    
    def test_process_unsupported_file(self):
        """测试处理不支持的文件格式"""
        result = self.handler.process_uploaded_file(b"content", "resume.jpg")
        
        self.assertFalse(result['success'])
        self.assertIsNotNone(result['error'])
        self.assertIn("不支持的文件格式", result['error'])


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        from app.nlp.text_processor import TextProcessor
        from app.nlp.similarity import SimilarityCalculator
        
        # 初始化组件
        processor = TextProcessor()
        calculator = SimilarityCalculator()
        
        # 测试数据
        resume_text = """
        John Smith
        Senior Python Developer
        
        Experience:
        - 5 years of Python development
        - Django and Flask frameworks
        - PostgreSQL database
        - Docker containerization
        - Git version control
        
        Skills: Python, Django, Flask, PostgreSQL, Docker, Git, HTML, CSS
        """
        
        job_text = """
        Senior Python Developer Position
        
        Requirements:
        - 3+ years of Python experience
        - Django framework knowledge
        - Database experience (PostgreSQL preferred)
        - Docker experience
        - Git version control
        - HTML/CSS skills
        
        Nice to have:
        - React knowledge
        - AWS experience
        """
        
        # 处理文本
        resume_data = processor.preprocess_for_matching(resume_text)
        job_data = processor.preprocess_for_matching(job_text)
        
        # 计算相似度
        result = calculator.calculate_comprehensive_score(resume_data, job_data)
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn('overall_score', result)
        self.assertIn('percentage_score', result)
        self.assertGreater(result['percentage_score'], 50)  # 应该有较高匹配度
        
        # 验证技能匹配
        skill_analysis = result['skill_analysis']
        self.assertGreater(len(skill_analysis['matched_skills']), 0)
        self.assertIn('python', skill_analysis['matched_skills'])
        self.assertIn('django', skill_analysis['matched_skills'])


if __name__ == "__main__":
    unittest.main() 