
python logic_llm/reasoning_colloction/basemodel/Reclor_train_data.py

if [ $? -ne 0 ]; then
    echo "basemodel/Reclor_train_data.py failed"
    exit 1
fi

python FOLIO_train_data.py

if [ $? -ne 0 ]; then
    echo "FOLIO_train_data.py failed"
    exit 1
fi
python LogiQA_v2_train_data.py

if [ $? -ne 0 ]; then
    echo "LogiQA_v2_train_data.py failed"
    exit 1
fi
python Reclor_train_data.py

if [ $? -ne 0 ]; then
    echo "Reclor_train_data.py failed"
    exit 1
fi