# DLT-Hallucination-Measurement

## To Run the Code
Install required libs:
```shell
pip3 install accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7
```

First train the model using the following command.
```shell
python3 train.py
```

Then run and test the model.
```shell
python3 test.py
```