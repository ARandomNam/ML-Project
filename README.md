# Resume Match Score Predictor

一个基于 NLP 的轻量级工具，用于比较简历与职位描述的匹配度，并返回匹配分数以及关键词分析。

## 🚀 功能特性

- **智能匹配**: 使用 TF-IDF + 余弦相似度和 spaCy 词向量进行文本相似度计算
- **关键词分析**: 自动提取和高亮匹配/缺失的技能关键词
- **多格式支持**: 支持文本直接输入和文件上传 (PDF, DOCX)
- **友好界面**: 基于 Streamlit 的简洁 web 界面
- **可扩展**: 模块化设计，便于未来升级到更复杂的 ML 模型

## 🛠 技术栈

- **Python**: 核心逻辑
- **spaCy + scikit-learn**: NLP 处理
- **Streamlit**: Web 界面
- **GitHub Actions**: CI/CD

## 📦 安装与运行

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

# 5. 运行应用
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

1. 访问 web 界面
2. 上传简历文件或粘贴文本
3. 输入职位描述
4. 查看匹配分数和关键词分析结果

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## �� 许可证

MIT License
