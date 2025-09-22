#!/bin/bash

# 运行第一个 Python 文件
python LogiQA_dev.py

# 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "LogiQA_dev.py 执行失败"
    exit 1
fi

# 运行第二个 Python 文件
python Reclor_dev.py

# 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "Reclor_dev.py 执行失败"
    exit 1
fi

# 运行第三个 Python 文件
python FOLIO_dev.py

# 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "FOLIO_dev.py 执行失败"
    exit 1
fi

# 运行第四个 Python 文件
python logiqa-zh_test.py

# 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "logiqa-zh_test.py 执行失败"
    exit 1
fi

