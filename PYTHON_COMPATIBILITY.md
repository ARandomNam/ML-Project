# Python 版本兼容性指南

## 📋 兼容性总结

| Python 版本 | 基础功能 | spaCy | Sentence-BERT | TensorFlow | 推荐度     |
| ----------- | -------- | ----- | ------------- | ---------- | ---------- |
| 3.8-3.10    | ✅       | ✅    | ✅            | ✅         | ⭐⭐⭐     |
| 3.11        | ✅       | ✅    | ✅            | ✅         | ⭐⭐⭐⭐⭐ |
| 3.12        | ✅       | ✅    | ✅            | ✅         | ⭐⭐⭐⭐⭐ |
| 3.13.2+     | ✅       | ✅    | ✅            | ❌         | ⭐⭐⭐⭐   |

## 🎯 当前测试环境

**您当前使用的是 Python 3.13.2**，经过测试确认：

### ✅ 完全可用的功能

- **基础 NLP 功能**: TF-IDF、spaCy 词向量、关键词匹配
- **🚀 深度学习功能**: Sentence-BERT、Transformers
- **Web 界面**: Streamlit 应用完全正常
- **文件处理**: PDF、DOCX、TXT 支持
- **可视化**: Plotly 图表和仪表盘

### ❌ 不可用的功能

- **TensorFlow**: 目前不支持 Python 3.13
  - 不影响核心功能
  - Sentence-BERT 和 PyTorch 提供完整的深度学习支持

## 📊 功能对比

### Python 3.13.2 (当前)

```
✅ 传统 NLP: TF-IDF + spaCy
✅ 深度学习: PyTorch + Sentence-BERT
✅ 高级模型: BERT、RoBERTa、MPNet
✅ GPU 支持: CUDA 自动检测
✅ Web 界面: 现代化 Streamlit UI
❌ TensorFlow: 暂不支持
```

### Python 3.11/3.12 (推荐)

```
✅ 传统 NLP: TF-IDF + spaCy
✅ 深度学习: PyTorch + Sentence-BERT + TensorFlow
✅ 高级模型: BERT、RoBERTa、MPNet + TensorFlow 模型
✅ GPU 支持: CUDA 自动检测
✅ Web 界面: 现代化 Streamlit UI
✅ TensorFlow: 完整支持
```

## 💡 建议

### 如果您使用 Python 3.13.2

1. **继续使用**: 功能已经非常完整
2. **安装命令**: `pip install -r requirements-python313.txt`
3. **优势**:
   - 最新 Python 特性和性能优化
   - 核心深度学习功能完全可用
   - Sentence-BERT 提供最先进的语义理解

### 如果需要 TensorFlow

1. **降级到 Python 3.11 或 3.12**
2. **安装命令**: `pip install -r requirements.txt`
3. **优势**:
   - 完整的深度学习生态
   - 更多预训练模型选择

## 🚀 性能对比

基于实际测试，Python 3.13.2 上的应用性能：

| 功能          | 性能    | 说明               |
| ------------- | ------- | ------------------ |
| 文本预处理    | 🚀 更快 | Python 3.13 优化   |
| TF-IDF 计算   | ✅ 正常 | scikit-learn 兼容  |
| spaCy 分析    | ✅ 正常 | 完全支持           |
| Sentence-BERT | ✅ 正常 | PyTorch 2.7+ 支持  |
| Web 界面      | ✅ 正常 | Streamlit 最新版本 |

## 🔧 故障排除

### 常见问题

1. **NLTK SSL 错误**

   ```python
   import ssl
   ssl._create_default_https_context = ssl._create_unverified_context
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

2. **依赖冲突**

   ```bash
   pip install --upgrade pip
   pip install -r requirements-python313.txt
   ```

3. **模型下载慢**
   - 首次运行会下载 Sentence-BERT 模型
   - 大约需要 500MB 存储空间
   - 使用国内镜像可能会更快

## 📞 支持

如果遇到任何问题：

1. 运行 `python3 check_setup.py` 诊断环境
2. 查看具体错误信息
3. 考虑是否需要切换 Python 版本
