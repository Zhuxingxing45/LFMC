# # #!/bin/bash

# 运行第一个 Python 文件
python fintuing_LogicQA_v2.py

# 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "fintuing_LogicQA_v2.py 执行失败"
    exit 1
fi




# # 运行第四个 Python 文件
# python fintuing_logiqa-zh_test.py

# # 检查上一个命令是否成功执行
# if [ $? -ne 0 ]; then
#     echo "fintuing_logiqa-zh_test.py 执行失败"
#     exit 1
# fi

# # 运行第三个 Python 文件
# python fintuing_FOLIO_dev.py

# # 检查上一个命令是否成功执行
# if [ $? -ne 0 ]; then
#     echo "fintuing_FOLIO_dev.py 执行失败"
#     exit 1
# fi

# # 运行第二个 Python 文件
# python fintuing_Reclor_dev.py

# # 检查上一个命令是否成功执行
# if [ $? -ne 0 ]; then
#     echo "fintuing_Reclor_dev.py 执行失败"
#     exit 1
# fi

