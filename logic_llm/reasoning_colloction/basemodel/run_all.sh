# # 运行第一个 Python 文件
# python FOLIO_train_data.py

# # 检查上一个命令是否成功执行
# if [ $? -ne 0 ]; then
#     echo "FOLIO_train_data.py 执行失败"
#     exit 1
# fi

# 运行第二个 Python 文件
python LogiQA_v2_train_data.py

# 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "LogiQA_v2_train_data.py 执行失败"
    exit 1
fi

# 运行第三个 Python 文件
python Reclor_train_data.py

# 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "Reclor_train_data.py 执行失败"
    exit 1
fi