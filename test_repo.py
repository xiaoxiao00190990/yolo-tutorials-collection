#!/usr/bin/env python3
"""
测试脚本 - 验证YOLO教程仓库的基本功能
"""

import os
import sys

def check_files():
    """检查必需文件是否存在"""
    required_files = [
        'README.md',
        'QUICKSTART.md',
        'requirements.txt',
        'train_example.py',
        'deploy_example.py',
        'LICENSE',
        '.gitignore'
    ]
    
    print("📁 检查文件结构...")
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (缺失)")
            missing_files.append(file)
    
    return len(missing_files) == 0

def check_readme_content():
    """检查README内容"""
    print("\n📖 检查README内容...")
    
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键部分
        sections = [
            '训练教程',
            '部署教程', 
            '中文资源',
            '快速开始'
        ]
        
        for section in sections:
            if section in content:
                print(f"  ✅ 包含 '{section}' 部分")
            else:
                print(f"  ⚠️  缺少 '{section}' 部分")
        
        return True
    except Exception as e:
        print(f"  ❌ 读取README失败: {e}")
        return False

def check_python_scripts():
    """检查Python脚本语法"""
    print("\n🐍 检查Python脚本语法...")
    
    scripts = ['train_example.py', 'deploy_example.py']
    all_valid = True
    
    for script in scripts:
        if os.path.exists(script):
            try:
                # 尝试解析语法
                with open(script, 'r', encoding='utf-8') as f:
                    compile(f.read(), script, 'exec')
                print(f"  ✅ {script} 语法正确")
            except SyntaxError as e:
                print(f"  ❌ {script} 语法错误: {e}")
                all_valid = False
        else:
            print(f"  ❌ {script} 不存在")
            all_valid = False
    
    return all_valid

def generate_summary():
    """生成项目摘要"""
    print("\n📊 项目摘要:")
    
    # 统计文件
    total_files = 0
    total_size = 0
    
    for root, dirs, files in os.walk('.'):
        # 忽略.git目录
        if '.git' in root:
            continue
            
        for file in files:
            if file.endswith(('.py', '.md', '.txt', '.yaml', '.yml')):
                filepath = os.path.join(root, file)
                total_files += 1
                total_size += os.path.getsize(filepath)
    
    print(f"  文件数量: {total_files}")
    print(f"  总大小: {total_size / 1024:.1f} KB")
    
    # 读取README第一行
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            first_line = f.readline().strip('# \n')
        print(f"  项目标题: {first_line}")
    except:
        pass
    
    # GitHub仓库信息
    print(f"  GitHub仓库: https://github.com/xiaoxiao00190990/yolo-tutorials-collection")

def main():
    print("=" * 60)
    print("YOLO教程仓库测试脚本")
    print("=" * 60)
    
    # 检查当前目录
    if not os.path.exists('README.md'):
        print("❌ 错误: 请在项目根目录运行此脚本")
        return 1
    
    # 执行检查
    files_ok = check_files()
    readme_ok = check_readme_content()
    scripts_ok = check_python_scripts()
    
    # 生成摘要
    generate_summary()
    
    # 总体结果
    print("\n" + "=" * 60)
    print("测试结果:")
    
    if files_ok and readme_ok and scripts_ok:
        print("✅ 所有检查通过! 仓库准备就绪。")
        print("\n🎉 下一步:")
        print("1. 访问GitHub仓库查看内容")
        print("2. 按照QUICKSTART.md开始使用")
        print("3. 分享给需要YOLO教程的朋友")
        return 0
    else:
        print("⚠️  部分检查未通过，请修复问题。")
        return 1

if __name__ == '__main__':
    sys.exit(main())