# Resume Match Score Predictor

一个基于 NLP 的轻量级工具，用于比较简历与职位描述的匹配度，并返回匹配分数以及关键词分析。

## 🚀 功能特性

- **🧠 多层次智能匹配**:
  - 传统 NLP: TF-IDF + 余弦相似度、spaCy 词向量
  - 🚀 **PyTorch 深度学习**: Sentence-BERT 和 Transformer 模型语义理解
- **⚡ GPU 加速**: 自动检测和使用 CUDA/MPS 加速深度学习计算 (支持 NVIDIA 和 Apple Silicon)
- **🔍 智能关键词分析**: 自动提取和高亮匹配/缺失的技能关键词
- **📁 多格式支持**: 支持文本直接输入和文件上传 (PDF, DOCX, TXT)
- **🎛️ 灵活配置**: 用户可调整传统和深度学习算法权重
- **📊 友好界面**: 基于 Streamlit 的现代化 web 界面
- **💡 智能回退**: 深度学习不可用时自动使用传统方法
- **🔧 高度可扩展**: 模块化设计，便于添加新功能

## 🛠 技术栈

### 核心技术

- **Python 3.8+**: 核心开发语言
- **spaCy + scikit-learn**: 传统 NLP 处理
- **NLTK**: 文本预处理

### 🚀 深度学习技术

- **PyTorch 1.12+**: 主要深度学习框架
- **Sentence-Transformers**: 预训练语义模型
- **Transformers (Hugging Face)**: 预训练语言模型
- **CUDA/MPS 支持**: GPU 加速（NVIDIA 和 Apple Silicon）

### Web 和部署

- **Streamlit**: 现代化 Web 界面
- **Plotly**: 交互式数据可视化
- **GitHub Actions**: CI/CD 自动化

## 📦 安装与运行

### ⚠️ Python 版本兼容性

- **推荐**: Python 3.11 或 3.12 （完整功能支持）
- **支持**: Python 3.13.2+ （除 TensorFlow 外的所有功能）
- **最低**: Python 3.8+

### 方法一：快速启动（推荐）

```bash
# 克隆仓库
git clone <repository-url>
cd resume-matcher

# 运行启动脚本（自动创建虚拟环境并安装依赖）
./run.sh
```

### 方法二：手动安装

```bash
# 1. 克隆仓库
git clone <repository-url>
cd resume-matcher

# 2. 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 4. 下载spaCy语言模型
python3 -m spacy download en_core_web_md

# 5. (可选) 安装PyTorch深度学习依赖以启用高级功能
pip install torch sentence-transformers transformers

# 6. 检查环境设置
python3 check_setup.py

# 7. 运行应用
streamlit run app/main.py
```

应用将在浏览器中自动打开，默认地址：`http://localhost:8501`

## 📁 项目结构

```
resume-matcher/
├── app/
│   ├── main.py              # Streamlit主应用
│   ├── nlp/                 # NLP核心模块
│   └── utils/               # 工具函数
├── tests/                   # 测试文件
├── data/sample/             # 示例数据
└── requirements.txt         # 依赖列表
```

## 🔧 使用方法

1. **环境检查**: 运行 `python3 check_setup.py` 检查依赖安装状态
2. **启动应用**: 访问 web 界面 (http://localhost:8501)
3. **选择分析方法**:
   - ✅ 传统 NLP 方法 (TF-IDF, spaCy)
   - 🚀 PyTorch 深度学习方法 (Sentence-BERT, BERT) - 需要额外依赖
4. **输入数据**: 上传简历文件或粘贴文本，输入职位描述
5. **调整权重**: 根据需要调整各算法权重
6. **查看结果**: 获得匹配分数、关键词分析和详细报告

### 💡 使用提示

- **GPU 加速**:
  - NVIDIA GPU: 自动使用 CUDA 加速
  - Apple Silicon (M1/M2/M3): 自动使用 MPS 加速
- **智能回退**: 即使 PyTorch 深度学习依赖缺失，应用仍可正常运行传统功能
- **权重调整**: 可根据具体需求调整不同算法的重要性
- **多模型融合**: PyTorch 版本支持多个 BERT 模型的加权融合，提供更准确的评估

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## �� 许可证

MIT License
