cd logic_llm/wrong_reasoning_colloction/basemodel

python FOLIO_train_data.py

if [ $? -ne 0 ]; then
    echo "FOLIO_train_data.py falied"
    exit 1
fi

python LogiQA_v2_train_data.py

if [ $? -ne 0 ]; then
    echo "LogiQA_v2_train_data.py falied"
    exit 1
fi

python Reclor_train_data.py

if [ $? -ne 0 ]; then
    echo "Reclor_train_data.py falied"
    exit 1
fi

python logiqa-zh_train_data.py

if [ $? -ne 0 ]; then
    echo "logiqa-zh_train_data.py falied"
    exit 1
fi