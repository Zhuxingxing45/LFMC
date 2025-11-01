#!/bin/bash

python LogiQA_dev.py

if [ $? -ne 0 ]; then
    echo "LogiQA_dev.py falied"
    exit 1
fi

python Reclor_dev.py

if [ $? -ne 0 ]; then
    echo "Reclor_dev.py falied"
    exit 1
fi

python FOLIO_dev.py

if [ $? -ne 0 ]; then
    echo "FOLIO_dev.py falied"
    exit 1
fi

python logiqa-zh_test.py


if [ $? -ne 0 ]; then
    echo "logiqa-zh_test.py falied"
    exit 1
fi

