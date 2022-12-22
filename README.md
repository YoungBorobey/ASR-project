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
   -c default_test_config.json \
   -r default_test_model/checkpoint.pth \
   -t test_data \
   -o test_result.json
```
### BPE tokenizer model usage
```shell
python test.py \
   -c default_test_config.json \
   -r default_test_model/checkpoint.pth \
   -t test_data \
   -o test_result.json
```
