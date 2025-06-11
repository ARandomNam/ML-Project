"""
PyTorch深度学习模块测试
"""

import unittest
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.nlp.deep_learning import PyTorchSimilarityCalculator, PYTORCH_AVAILABLE
    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False


@unittest.skipUnless(IMPORT_SUCCESS, "PyTorch dependencies not available")
class TestPyTorchSimilarityCalculator(unittest.TestCase):
    """PyTorch深度学习计算器测试"""
    
    def setUp(self):
        if PYTORCH_AVAILABLE:
            self.calculator = PyTorchSimilarityCalculator()
        else:
            self.calculator = None
    
    @unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch not available")
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.calculator)
        self.assertTrue(hasattr(self.calculator, 'available'))
        self.assertTrue(hasattr(self.calculator, 'models'))
        self.assertTrue(hasattr(self.calculator, 'device'))
    
    @unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch not available")
    def test_device_detection(self):
        """测试设备检测"""
        device = self.calculator._get_device()
        self.assertIn(device, ['cuda', 'mps', 'cpu', None])
    
    @unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch not available")
    def test_model_info(self):
        """测试模型信息获取"""
        info = self.calculator.get_model_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('available', info)
        self.assertIn('device', info)
        self.assertIn('pytorch_version', info)
        self.assertIn('models_loaded', info)
    
    @unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch not available")
    def test_sentence_bert_similarity(self):
        """测试Sentence-BERT相似度计算"""
        if not self.calculator.available:
            self.skipTest("PyTorch models not available")
        
        text1 = "Python developer with Django experience"
        text2 = "Looking for Python Django developer"
        
        similarity = self.calculator.calculate_sentence_bert_similarity(text1, text2)
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    @unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch not available")
    def test_bert_similarity(self):
        """测试BERT相似度计算"""
        if not self.calculator.available:
            self.skipTest("PyTorch models not available")
        
        text1 = "Machine learning engineer"
        text2 = "AI specialist position"
        
        similarity = self.calculator.calculate_bert_similarity(text1, text2)
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    @unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch not available")
    def test_weighted_similarity(self):
        """测试加权多模型相似度"""
        if not self.calculator.available:
            self.skipTest("PyTorch models not available")
        
        text1 = "Data scientist with Python skills"
        text2 = "Python data analyst role"
        
        result = self.calculator.calculate_weighted_similarity(text1, text2)
        
        self.assertIsInstance(result, dict)
        self.assertIn('weighted_average', result)
        self.assertGreaterEqual(result['weighted_average'], 0.0)
        self.assertLessEqual(result['weighted_average'], 1.0)
    
    @unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch not available")
    def test_enhanced_similarity(self):
        """测试增强相似度计算"""
        if not self.calculator.available:
            self.skipTest("PyTorch models not available")
        
        resume_text = """
        Senior Python Developer
        5 years experience with Django, Flask, PostgreSQL
        Strong background in web development and API design
        """
        
        job_text = """
        Python Developer Position
        Requirements: 3+ years Python, Django experience
        PostgreSQL database knowledge preferred
        """
        
        result = self.calculator.calculate_enhanced_similarity(resume_text, job_text)
        
        self.assertIsInstance(result, dict)
        self.assertIn('overall_score', result)
        self.assertIn('multi_model_scores', result)
        self.assertIn('device_used', result)
        
        if 'error' not in result:
            self.assertGreaterEqual(result['overall_score'], 0.0)
            self.assertLessEqual(result['overall_score'], 1.0)
    
    @unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch not available")
    def test_attention_keywords_extraction(self):
        """测试注意力机制关键词提取"""
        if not self.calculator.available:
            self.skipTest("PyTorch models not available")
        
        text = """
        Experienced software engineer with expertise in Python programming.
        Strong background in machine learning and data science.
        Proficient in Django web framework and PostgreSQL database.
        """
        
        keywords = self.calculator.extract_attention_keywords(text)
        
        self.assertIsInstance(keywords, list)
        # 检查返回的是元组格式 (keyword, weight)
        if len(keywords) > 0:
            self.assertTrue(all(isinstance(kw, tuple) and len(kw) == 2 for kw in keywords))
            self.assertTrue(all(isinstance(kw[0], str) and isinstance(kw[1], float) for kw in keywords))
    
    @unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch not available")
    def test_batch_similarity(self):
        """测试批量相似度计算"""
        if not self.calculator.available:
            self.skipTest("PyTorch models not available")
        
        texts1 = ["Python developer", "Data scientist", "Web developer"]
        texts2 = ["Django developer", "ML engineer", "Frontend developer"]
        
        similarities = self.calculator.calculate_semantic_similarity_batch(texts1, texts2)
        
        self.assertIsInstance(similarities, list)
        self.assertEqual(len(similarities), len(texts1))
        self.assertTrue(all(0.0 <= sim <= 1.0 for sim in similarities))
    
    def test_fallback_behavior(self):
        """测试回退行为（当PyTorch不可用时）"""
        if not PYTORCH_AVAILABLE:
            # 测试导入失败的情况
            self.assertFalse(IMPORT_SUCCESS)
        else:
            # 如果可用，测试正常行为
            self.assertTrue(IMPORT_SUCCESS)


class TestIntegrationWithSimilarityCalculator(unittest.TestCase):
    """集成测试：PyTorch深度学习与相似度计算器的集成"""
    
    def test_similarity_calculator_with_pytorch(self):
        """测试相似度计算器的PyTorch深度学习集成"""
        try:
            from app.nlp.similarity import SimilarityCalculator
            
            calculator = SimilarityCalculator()
            
            # 检查深度学习是否可用
            has_dl = hasattr(calculator, 'deep_learning_available')
            
            if has_dl and calculator.deep_learning_available:
                # 测试PyTorch深度学习功能
                resume_data = {
                    'cleaned': 'python developer django experience',
                    'keywords': ['python', 'developer', 'django'],
                    'tech_skills': ['python', 'django']
                }
                
                job_data = {
                    'cleaned': 'python django developer position',
                    'keywords': ['python', 'django', 'developer'],
                    'tech_skills': ['python', 'django']
                }
                
                result = calculator.calculate_comprehensive_score(resume_data, job_data)
                
                # 检查是否包含PyTorch深度学习结果
                self.assertIn('deep_learning_available', result)
                
                if result['deep_learning_available']:
                    self.assertIn('pytorch_similarity', result['scores'])
                    
                    # 检查PyTorch结果详情
                    if 'pytorch_results' in result:
                        pytorch_results = result['pytorch_results']
                        self.assertIn('overall_score', pytorch_results)
                        self.assertIn('multi_model_scores', pytorch_results)
            else:
                # 如果深度学习不可用，确保系统仍能正常工作
                self.assertIsInstance(calculator, SimilarityCalculator)
        
        except Exception as e:
            self.fail(f"Integration test failed: {e}")


class TestBackwardCompatibility(unittest.TestCase):
    """向后兼容性测试"""
    
    def test_deep_learning_calculator_alias(self):
        """测试DeepLearningCalculator别名是否正常工作"""
        try:
            from app.nlp.deep_learning import DeepLearningCalculator
            
            # 应该与PyTorchSimilarityCalculator是同一个类
            if PYTORCH_AVAILABLE:
                calculator = DeepLearningCalculator()
                self.assertIsInstance(calculator, PyTorchSimilarityCalculator)
            
        except ImportError:
            if PYTORCH_AVAILABLE:
                self.fail("DeepLearningCalculator alias should be available when PyTorch is available")


if __name__ == "__main__":
    unittest.main() 