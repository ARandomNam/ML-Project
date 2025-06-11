#!/bin/bash

# Resume Match Score Predictor 启动脚本

echo "🎯 Resume Match Score Predictor"
echo "================================"

# 检查Python版本
echo "检查Python环境..."
python3 --version

# 检查是否存在虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "安装依赖包..."
pip install --upgrade pip
pip install -r requirements.txt

# 下载spaCy语言模型
echo "下载spaCy语言模型..."
python3 -m spacy download en_core_web_md

# 下载深度学习模型（可选）
echo "下载深度学习模型（这可能需要几分钟）..."
python3 -c "
try:
    from sentence_transformers import SentenceTransformer
    print('正在下载 Sentence-BERT 模型...')
    SentenceTransformer('all-MiniLM-L6-v2')
    print('✅ 深度学习模型下载完成')
except Exception as e:
    print(f'⚠️ 深度学习模型下载失败: {e}')
    print('应用仍可正常运行，但将不包含深度学习功能')
"

# 运行应用
echo "启动Streamlit应用..."
echo "应用将在浏览器中打开: http://localhost:8501"
echo "🚀 深度学习功能已启用！"
streamlit run app/main.py 