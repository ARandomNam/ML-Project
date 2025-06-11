"""
PyTorch深度学习相似度计算模块
使用PyTorch和预训练的BERT、RoBERTa等transformer模型进行语义相似度计算
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    PYTORCH_AVAILABLE = True
    
except ImportError as e:
    print(f"PyTorch深度学习依赖未安装: {e}")
    PYTORCH_AVAILABLE = False


class PyTorchSimilarityCalculator:
    """PyTorch深度学习相似度计算器"""
    
    def __init__(self):
        self.available = PYTORCH_AVAILABLE
        self.models = {}
        self.device = self._get_device()
        
        if self.available:
            self._initialize_models()
    
    def _get_device(self):
        """获取最佳计算设备"""
        if not PYTORCH_AVAILABLE:
            return None
        
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"✅ 使用GPU加速: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'  # Apple Silicon GPU
            print("✅ 使用Apple Silicon GPU加速")
        else:
            device = 'cpu'
            print("⚠️ 使用CPU计算")
        
        return device
    
    def _initialize_models(self):
        """初始化PyTorch预训练模型"""
        try:
            print("正在加载PyTorch深度学习模型...")
            
            # Sentence-BERT 模型 (推荐用于语义相似度)
            self.models['sentence_bert'] = SentenceTransformer('all-MiniLM-L6-v2')
            self.models['sentence_bert'].to(self.device)
            print("✅ Sentence-BERT (MiniLM) 模型加载完成")
            
            # 更大的多语言模型
            try:
                self.models['sentence_bert_large'] = SentenceTransformer('all-mpnet-base-v2')
                self.models['sentence_bert_large'].to(self.device)
                print("✅ Sentence-BERT (MPNet) 大模型加载完成")
            except Exception as e:
                print(f"⚠️ 大模型加载失败: {e}")
            
            # 专用的BERT模型用于细粒度控制
            try:
                self.models['bert_base'] = {
                    'tokenizer': AutoTokenizer.from_pretrained('bert-base-uncased'),
                    'model': AutoModel.from_pretrained('bert-base-uncased').to(self.device)
                }
                print("✅ BERT-base 模型加载完成")
            except Exception as e:
                print(f"⚠️ BERT模型加载失败: {e}")
            
        except Exception as e:
            print(f"❌ PyTorch深度学习模型加载失败: {e}")
            self.available = False
    
    def calculate_sentence_bert_similarity(self, text1: str, text2: str, 
                                         model_name: str = 'sentence_bert') -> float:
        """
        使用Sentence-BERT计算语义相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            model_name: 使用的模型名称
            
        Returns:
            相似度分数 (0-1)
        """
        if not self.available or model_name not in self.models:
            return 0.0
        
        try:
            model = self.models[model_name]
            
            # 生成句子嵌入
            with torch.no_grad():
                embeddings = model.encode([text1, text2], 
                                        convert_to_tensor=True,
                                        device=self.device)
                
                # 计算余弦相似度
                similarity = F.cosine_similarity(embeddings[0:1], embeddings[1:2])
                
            return float(torch.clamp(similarity, 0.0, 1.0).item())
            
        except Exception as e:
            print(f"Sentence-BERT计算错误: {e}")
            return 0.0
    
    def calculate_bert_similarity(self, text1: str, text2: str) -> float:
        """
        使用原生BERT模型计算相似度，提供更多控制
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度分数 (0-1)
        """
        if not self.available or 'bert_base' not in self.models:
            return 0.0
        
        try:
            tokenizer = self.models['bert_base']['tokenizer']
            model = self.models['bert_base']['model']
            
            # 编码文本
            inputs1 = tokenizer(text1, return_tensors='pt', truncation=True, 
                              padding=True, max_length=512).to(self.device)
            inputs2 = tokenizer(text2, return_tensors='pt', truncation=True, 
                              padding=True, max_length=512).to(self.device)
            
            # 获取嵌入
            with torch.no_grad():
                outputs1 = model(**inputs1)
                outputs2 = model(**inputs2)
                
                # 使用平均池化获得句子嵌入
                embedding1 = outputs1.last_hidden_state.mean(dim=1)
                embedding2 = outputs2.last_hidden_state.mean(dim=1)
                
                # 计算余弦相似度
                similarity = F.cosine_similarity(embedding1, embedding2)
                
            return float(torch.clamp(similarity, 0.0, 1.0).item())
            
        except Exception as e:
            print(f"BERT计算错误: {e}")
            return 0.0
    
    def calculate_weighted_similarity(self, text1: str, text2: str, 
                                    weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        计算加权多模型相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            weights: 各模型权重
            
        Returns:
            包含各模型得分和加权总分的字典
        """
        if weights is None:
            weights = {
                'sentence_bert': 0.4,
                'sentence_bert_large': 0.4,
                'bert_base': 0.2
            }
        
        results = {}
        total_weight = 0.0
        weighted_sum = 0.0
        
        # Sentence-BERT
        if 'sentence_bert' in self.models and 'sentence_bert' in weights:
            score = self.calculate_sentence_bert_similarity(text1, text2, 'sentence_bert')
            results['sentence_bert'] = score
            weighted_sum += score * weights['sentence_bert']
            total_weight += weights['sentence_bert']
        
        # Large Sentence-BERT
        if 'sentence_bert_large' in self.models and 'sentence_bert_large' in weights:
            score = self.calculate_sentence_bert_similarity(text1, text2, 'sentence_bert_large')
            results['sentence_bert_large'] = score
            weighted_sum += score * weights['sentence_bert_large']
            total_weight += weights['sentence_bert_large']
        
        # BERT
        if 'bert_base' in self.models and 'bert_base' in weights:
            score = self.calculate_bert_similarity(text1, text2)
            results['bert_base'] = score
            weighted_sum += score * weights['bert_base']
            total_weight += weights['bert_base']
        
        # 计算加权平均
        results['weighted_average'] = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return results
    
    def calculate_semantic_similarity_batch(self, texts1: List[str], 
                                          texts2: List[str],
                                          batch_size: int = 32) -> List[float]:
        """
        批量计算语义相似度，支持批处理优化
        
        Args:
            texts1: 第一组文本列表
            texts2: 第二组文本列表
            batch_size: 批处理大小
            
        Returns:
            相似度分数列表
        """
        if not self.available or 'sentence_bert' not in self.models:
            return [0.0] * len(texts1)
        
        try:
            model = self.models['sentence_bert']
            similarities = []
            
            # 分批处理
            for i in range(0, len(texts1), batch_size):
                batch_texts1 = texts1[i:i+batch_size]
                batch_texts2 = texts2[i:i+batch_size]
                
                with torch.no_grad():
                    # 批量编码
                    embeddings1 = model.encode(batch_texts1, 
                                             convert_to_tensor=True,
                                             device=self.device)
                    embeddings2 = model.encode(batch_texts2, 
                                             convert_to_tensor=True,
                                             device=self.device)
                    
                    # 批量计算相似度
                    batch_similarities = F.cosine_similarity(embeddings1, embeddings2)
                    batch_similarities = torch.clamp(batch_similarities, 0.0, 1.0)
                    
                    similarities.extend(batch_similarities.cpu().numpy().tolist())
            
            return similarities
            
        except Exception as e:
            print(f"批量计算错误: {e}")
            return [0.0] * len(texts1)
    
    def extract_attention_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        使用注意力机制提取关键词
        
        Args:
            text: 输入文本
            top_k: 返回的关键词数量
            
        Returns:
            关键词及其权重的列表
        """
        if not self.available or 'bert_base' not in self.models:
            return []
        
        try:
            tokenizer = self.models['bert_base']['tokenizer']
            model = self.models['bert_base']['model']
            
            # 编码文本
            inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                             padding=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
                
                # 获取注意力权重 (最后一层，所有头的平均)
                attentions = outputs.attentions[-1]  # 最后一层
                avg_attention = attentions.mean(dim=1).squeeze()  # 平均所有头
                
                # 获取token及其注意力权重
                tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
                token_weights = avg_attention.mean(dim=0)  # 平均所有位置
                
                # 过滤特殊token并排序
                keyword_weights = []
                for token, weight in zip(tokens, token_weights):
                    if token not in ['[CLS]', '[SEP]', '[PAD]'] and not token.startswith('##'):
                        keyword_weights.append((token, float(weight.item())))
                
                # 按权重排序并返回top_k
                keyword_weights.sort(key=lambda x: x[1], reverse=True)
                return keyword_weights[:top_k]
            
        except Exception as e:
            print(f"注意力关键词提取错误: {e}")
            return []
    
    def calculate_enhanced_similarity(self, resume_text: str, job_text: str) -> Dict[str, Any]:
        """
        计算增强的相似度分析
        
        Args:
            resume_text: 简历文本
            job_text: 工作描述文本
            
        Returns:
            详细的相似度分析结果
        """
        if not self.available:
            return {'overall_score': 0.0, 'error': 'PyTorch深度学习不可用'}
        
        try:
            # 多模型相似度计算
            multi_scores = self.calculate_weighted_similarity(resume_text, job_text)
            
            # 提取关键词
            resume_keywords = self.extract_attention_keywords(resume_text, 15)
            job_keywords = self.extract_attention_keywords(job_text, 15)
            
            # 计算关键词匹配
            resume_keyword_set = set([kw[0].lower() for kw in resume_keywords])
            job_keyword_set = set([kw[0].lower() for kw in job_keywords])
            common_keywords = resume_keyword_set.intersection(job_keyword_set)
            
            keyword_match_score = len(common_keywords) / max(len(job_keyword_set), 1)
            
            # 综合评分
            overall_score = (
                multi_scores.get('weighted_average', 0.0) * 0.7 +
                keyword_match_score * 0.3
            )
            
            return {
                'overall_score': overall_score,
                'multi_model_scores': multi_scores,
                'keyword_match_score': keyword_match_score,
                'resume_keywords': resume_keywords,
                'job_keywords': job_keywords,
                'common_keywords': list(common_keywords),
                'device_used': self.device
            }
            
        except Exception as e:
            print(f"增强相似度计算错误: {e}")
            return {'overall_score': 0.0, 'error': str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.available:
            return {'available': False, 'error': 'PyTorch不可用'}
        
        info = {
            'available': True,
            'device': self.device,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'models_loaded': list(self.models.keys())
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
        
        return info


# 为了向后兼容，保留原来的类名
DeepLearningCalculator = PyTorchSimilarityCalculator 