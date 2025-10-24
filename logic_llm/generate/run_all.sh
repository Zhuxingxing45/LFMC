#!/bin/bash

# # 运行第一个 Python 文件
# python logicFintuing/fintuing_LogicQA_v2.py

# # 检查上一个命令是否成功执行
# if [ $? -ne 0 ]; then
#     echo "logicFintuing/fintuing_LogicQA_v2.py 执行失败"
#     exit 1
# fi

# 运行第二个 Python 文件
python logicFintuing/fintuing_Reclor_dev.py

# 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "logicFintuing/fintuing_Reclor_dev.py 执行失败"
    exit 1
fi

# 运行第三个 Python 文件
python logicotFintuing/fintuing_FOLIO_dev.py

# 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "logicotFintuing/fintuing_FOLIO_dev.py 执行失败"
    exit 1
fi

# 运行第四个 Python 文件
python logicotFintuing/fintuing_LogiQA_v2_dev.py

# 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "logicotFintuing/fintuing_LogiQA_v2_dev.py 执行失败"
    exit 1
fi

# 运行第五个 Python 文件
python logicotFintuing/fintuing_Reclor_dev.py

# 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "logicotFintuing/fintuing_Reclor_dev.py 执行失败"
    exit 1
fi


