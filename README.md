# Automatic Speech Recognition

## Installation guide

Clone this repository. Install required packages with line below
```shell
pip install -r ./requirements.txt
```

## Model usage
### Default model usage
```shell
python test.py \
   -c default_test_model/config.json \
   -r default_test_model/checkpoint.pth \
   -t test_data \
   -o test_result.json \
   -b 5
```
### BPE tokenizer model usage
```shell
python test.py \
   -c default_test_model_BPE/config.json \
   -r default_test_model_BPE/checkpoint.pth \
   -t test_data \
   -o test_result.json \
   -b 5 \
   -tok default_test_model_BPE/tokenizer_BPE_40.json
```
