Most of the code in this reposetory was taken from [here](https://github.com/talbaumel/RSAsummarization).

# Requirements
- Python 2.7
- CUDA 8.0
- cuDNN v5 (May 27, 2016), for CUDA 8.0
- tensorflow-gpu 1.2.1

# Installation
Using Anaconda:
```
# Clone the repository:
git clone https://github.com/stepgazaille/RSAsummarization.git
cd RSAsummarization

# Create and activate a virtual environment:
conda env create -f rsasum.yml
conda activate rsa

# Install tensorflow_gpu-1.2.1:
pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.1-cp27-none-linux_x86_64.whl

# Download NLTK stopwords:
python -m nltk.downloader stopwords

# Install pythonrouge
pip install git+https://github.com/tagucci/pythonrouge.git
```

Update "cmd" and "generated_path" and variables from multidocs.py:
```
# TODO: take the following as script parameters?
cmd = 'python run_summarization.py --mode=decode --single_pass=1 --coverage=True --vocab_path=../data/DMQA/finished_files/vocab --log_root=log --exp_name=myexperiment --data_path=test/temp_file --max_enc_steps=4000'
generated_path = 'log/myexperiment/decode_test_4000maxenc_4beam_35mindec_100maxdec_ckpt-238410/'
```


# Datasets
## Pretrained model
Download the Tensorflow 1.2.1 pretrained model from [here](https://github.com/abisee/pointer-generator) and copy it to the RSAsummarization directory:
```
cd RSAsummarization
mkdir test
mkdir -p log/myexperiment/

# Adapt those:
unzip ../data/RSAsum_models/pretrained_model_tf1.2.1.zip
cp -r ../data/RSAsum_models/pretrained_model_tf1.2.1/train/ log/myexperiment/train/
cp -r ../data/RSAsum_models/pretrained_model_tf1.2.1/decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/ log/myexperiment/decode_test_4000maxenc_4beam_35mindec_100maxdec_ckpt-238410/
```

## DMQA data
All you need is [here](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail).

## DUC data
Get permission [here](https://duc.nist.gov/data.html) and then:
```
# TODO: automate all this...
## 2005
mkdir -p duc05tokenized/testdata/
tar -zxvf DUC2005_Summarization_Documents.tgz
tar -zxvf DUC2005_Summarization_Documents/duc2005_docs.tar.gz
mv duc2005_docs/ duc05tokenized/testdata/docs/
rm -rf DUC2005_Summarization_Documents

## 2006
mkdir -p duc06tokenized/testdata/
tar -zxvf DUC2006_Summarization_Documents.tgz
tar -zxvf DUC2006_Summarization_Documents/duc2006_docs.tar.gz
mv duc2006_docs/ duc06tokenized/testdata/docs/
rm -rf DUC2006_Summarization_Documents

## 2007
mkdir -p duc07tokenized/testdata/
tar -zxvf DUC2007_Summarization_Documents.tgz
tar -zxvf DUC2007_Summarization_Documents/duc2007_testdocs.tar.gz
mv duc2007_testdocs/main/ duc07tokenized/testdata/docs/
rm -rf DUC2007_Summarization_Documents/ duc2007_testdocs/

# Extract models:
##2005
tar -xvf results.tar
mv results/ROUGE/models/ duc05tokenized/models/
rm -rf results/

## 2006:
tar -zxvf NISTeval.tar.gz
mv NISTeval/ROUGE/models/ duc06tokenized/models/
rm -rf NISTeval/


## 2007:
tar -zxvf mainEval.tar.gz
mv mainEval/ROUGE/models/ duc07tokenized/models/
rm -rf mainEval/

## DUC preproc:
# remove special character after word "warning" in file duc05tokenized/models/D400.M.250.B.J
# replace "é" characters with "e" in word "émigré" in file duc05tokenized/models/D354.M.250.C.C
# replace "é" characters with "e" word "Jean Chrétien" in ??
```


# Usage
Normal:
```
python runsum.py
# or
python multi_doc.py
# etc.
```
With logs:
```
python -u  runsum.py 2>&1 | tee log/$(date '+%Y-%m-%d-%Hh%Mm%Ss').log
# or
python -u  multi_doc.py 2>&1 | tee log/$(date '+%Y-%m-%d-%Hh%Mm%Ss').log
# etc.
```
