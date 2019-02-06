Most of the code in this reposetory was taken from [here](https://github.com/talbaumel/RSAsummarization).

# Requirements
NVIDIA:
- CUDA 8.0
- cuDNN v5

Java:
- Stanford CoreNLP

Perl:
- ROUGE-1.5.5


Python 2.7:
- tensorflow-gpu 1.2.1
- pyrouge


# Installation
## NVIDIA
- Install NVIDIA drivers from [here](https://www.nvidia.com/Download/index.aspx?lang=en-us).
- Download CUDA 8 from [here](https://developer.nvidia.com/cuda-90-download-archive), and install using [these instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
- Download cuDNN v5 (May 27, 2016), for CUDA 8.0 from [here](https://developer.nvidia.com/rdp/cudnn-archive), and install using [these instructions](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html).


## Java
Instructions to install the Stanford CoreNLP can by found [here](https://github.com/stanfordnlp/CoreNLP).


## Perl
Instructions to install ROUGE-1.5.5 can by found [here](https://web.archive.org/web/20171107220839/www.summarizerman.com/post/42675198985/figuring-out-rouge).



## Python
Create the project's virtual environment using Anaconda:
```
# Clone the repository:
git clone https://github.com/stepgazaille/RSAsummarization.git
cd RSAsummarization

# Create the virtual environment:
conda env create -f rsasum.yml

# Activate the environment with:
conda activate rsa
```
Download NLTK stopwords
```
python -m nltk.downloader stopwords
```
Install tensorflow_gpu-1.2.1:
```
pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.1-cp27-none-linux_x86_64.whl
```
Install pyrouge:
```
pip install git+https://github.com/tagucci/pythonrouge.git
pyrouge.test

# if "Cannot open exception db file for reading: rouge_installed_path/ROUGE-1.5.5/data/WordNet-2.0.exc.db" error:
cd pythonrouge/RELEASE-1.5.5/data/
rm WordNet-2.0.exc.db
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
```

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
# Datasets
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
```

### Pre-processing
For each of all three DUC data sets, execute the three preproc scripts and copy the resulting duc0Ntokenized directory to the RSAsummarization directory:
```
python duc200N_docs_preproc.py
python duc200N_models_preproc.py
python duc200N_queries_preproc.py
cp -R duc0Ntokenized/ RSAsummarization/duc0Ntokenized
```

# Usage
Open and execute one of the following Jupyter notebooks: runsum.ipynb, runvis.ipynb, d366i.ipynb or multi_doc.ipynb. You might need to edit the notebooks to use path to data according to your own setup.
