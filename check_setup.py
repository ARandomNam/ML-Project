#!/usr/bin/env python3
"""
Resume Match Score Predictor - 设置检查脚本
检查项目依赖和模块是否正确安装
"""

import sys
import importlib

def check_module(module_name, description=""):
    """检查模块是否可以导入"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} - {description}")
        return True
    except ImportError:
        print(f"❌ {module_name} - {description} (未安装)")
        return False

def main():
    print("🎯 Resume Match Score Predictor - 环境检查")
    print("=" * 50)
    
    # 检查 Python 版本
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python 版本过低: {python_version.major}.{python_version.minor}.{python_version.micro} (需要 >= 3.8)")
    
    print("\n📦 基础依赖检查:")
    basic_deps = [
        ("numpy", "数值计算"),
        ("pandas", "数据处理"),
        ("sklearn", "机器学习"),
        ("nltk", "自然语言处理"),
        ("spacy", "高级NLP"),
        ("streamlit", "Web框架"),
        ("plotly", "数据可视化"),
    ]
    
    basic_ok = True
    for module, desc in basic_deps:
        if not check_module(module, desc):
            basic_ok = False
    
    print("\n🚀 PyTorch深度学习依赖检查:")
    dl_deps = [
        ("torch", "PyTorch深度学习框架"),
        ("transformers", "Hugging Face Transformers"),
        ("sentence_transformers", "Sentence-BERT模型"),
    ]
    
    dl_ok = True
    for module, desc in dl_deps:
        if not check_module(module, desc):
            dl_ok = False
    
    # 检查实际深度学习功能可用性
    try:
        import os
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        from app.nlp.deep_learning import PyTorchSimilarityCalculator
        dl_calc = PyTorchSimilarityCalculator()
        actual_dl_available = dl_calc.available
        if actual_dl_available:
            print("✅ PyTorch深度学习功能实际可用")
            # 显示设备信息
            device_info = dl_calc.get_model_info()
            if device_info.get('available'):
                device = device_info.get('device', 'CPU')
                print(f"   设备: {device}")
                if device == 'cuda':
                    gpu_name = device_info.get('gpu_name', 'Unknown')
                    print(f"   GPU: {gpu_name}")
        else:
            print("❌ PyTorch深度学习功能实际不可用")
    except Exception as e:
        actual_dl_available = False
        print(f"❌ PyTorch深度学习模块导入失败: {e}")
    
    print("\n📄 文件处理依赖检查:")
    file_deps = [
        ("PyPDF2", "PDF文件处理"),
        ("docx", "Word文档处理"),
    ]
    
    file_ok = True
    for module, desc in file_deps:
        if not check_module(module, desc):
            file_ok = False
    
    print("\n" + "=" * 50)
    print("📊 总结:")
    
    if basic_ok:
        print("✅ 基础功能：完全可用")
    else:
        print("❌ 基础功能：缺少依赖")
        print("   安装命令: pip install -r requirements.txt")
    
    if dl_ok and actual_dl_available:
        print("✅ PyTorch深度学习功能：完全可用")
    else:
        print("❌ PyTorch深度学习功能：不可用")
        print("   安装命令: pip install torch sentence-transformers transformers")
    
    if file_ok:
        print("✅ 文件处理功能：完全可用")
    else:
        print("⚠️ 文件处理功能：部分缺失")
    
    print("\n🚀 启动建议:")
    if basic_ok:
        print("   可以运行: ./run.sh 或 streamlit run app/main.py")
    else:
        print("   请先安装依赖: pip install -r requirements.txt")
    
    print("\n💡 提示: 即使PyTorch深度学习依赖缺失，应用仍可正常运行传统NLP功能")

if __name__ == "__main__":
    main() 