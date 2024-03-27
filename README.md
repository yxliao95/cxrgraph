# Code for CXRGraph

## Trained model

CXRGraph:
Entity model: https://huggingface.co/yxLiao/biomedbert_ent_cxrgraph (related code: pipe1_ner_tokaux_attrcls_sent.py).
Relation model: https://huggingface.co/yxLiao/biomedbert_rel_cxrgraph (related code: pipe2_re_tokaux_sent.py).
Results: 
mimic: ner-f1=96.618%, attr-f1=91.996% re-f1=89.510%;
chexpert: ner-f1=96.059%, attr-f1=89.787% re-f1=86.643%

RadGraph:
Entity model: https://huggingface.co/yxLiao/biomedbert_ent_radgraph (related code: pipe1_ner_tokaux_sent.py).
Relation model: https://huggingface.co/yxLiao/biomedbert_rel_radgraph (related code: pipe2_re_tokaux_sent.py).
Results:
mimic: ner-f1=94.361%, attr-f1=85.202% re-f1=83.274%;
chexpert: ner-f1=91.228%, attr-f1=75.367% re-f1=73.302%

SciERC:
Entity model: https://huggingface.co/yxLiao/scibert_ent_scierc (related code: pipe1_ner_tokaux_sent.py).
Relation model: https://huggingface.co/yxLiao/scibert_rel_scierc (related code: pipe2_re_tokaux_sent.py).
Results: ner-f1=70.696%, re-f1=43.714%

ACE05:
Entity model: https://huggingface.co/yxLiao/bert_ent_ace05 (related code: pipe1_ner_tokaux_sent.py).
Relation model: https://huggingface.co/yxLiao/bert_rel_ace05 (related code: pipe2_re_tokaux_sent.py).
Results: ner-f1=90.299%, re-f1=67.140%

(For re-f1, entity boundries, entity types, and relation types must be all correct)

## Datasets

1. ACE05: We follow the instructions from [DyGIE](https://github.com/luanyi/DyGIE/tree/master/preprocessing) repo to preprocess the ACE05 dataset.
2. SciERC: We download the SciERC dataset from [Luan et al.](http://nlp.cs.washington.edu/sciIE/)
3. RadGraph: We download the RadGraph from [PhysioNet](https://physionet.org/content/radgraph/1.0.0/) and process the data by [this script](./preprocessing/radgraph2json.ipynb)
4. CXRGraph: Under the review process of PhysioNet. A link will be updated when available.

## Environments

### Device Details

Our models are trained on a cloud device rent from [AutoDL](https://www.autodl.com/)

- Mirror: PyTorch 1.11.0, Python 3.8(ubuntu20.04), Cuda 11.3
- GPU: RTX A5000(24GB) * 1, GPU Drive: 535.98
- CPU: 16 vCPU Intel(R) Xeon(R) Platinum 8350C CPU @ 2.60GHz
- Memory: 42GB

### Libraries

Conda install via:`conda env create -f environment.yml`

or manually install as follows (recommend):

1. `conda create --name cxr_graph python=3.9 -y`
   1. require python>=3.7 (requiring the build-in dict to keep insertion order)
2. `conda activate cxr_graph`
3. `conda install tqdm`
4. `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch`
   1. if you have compatiable issue to local cuda version, try the followings:
      1. `nvcc -V` to check cuda version
      2. `python` -> `import torch` -> `print(torch.__version__)` -> `exit()` to check torch+cuda version
      3. Find the correct torch version+cuda from [here](https://pytorch.org/get-started/previous-versions/)
   2. If you are using spacy, you may get the following errors:
      1. ImportError: cannot import name 'COMMON_SAFE_ASCII_CHARACTERS' from 'charset_normalizer.constant' (/root/miniconda3/envs/cxr_graph/lib/python3.9/site-packages/charset_normalizer/constant.py)
      2. `pip install chardet`
   3. TypeError: issubclass() arg 1 must be a class
      1. `pip install typing_extensions==4.4.0`
      2. See [link](https://github.com/explosion/spaCy/issues/12659)
      3. check version: `pip show spacy`
5. `pip install transformers==4.20.1`
6. `pip install notebook` (optional)


### Download HuggingFace models (if necessary)

1. Follows the [instruction](https://huggingface.co/docs/transformers/installation#offline-mode) to download.

## Training and Evaluation

Check the config.py for model configurations.
Full information are available in the `train.log` in the corresponding model download pages.

`pipe1_ner_tokaux_sent.py`: the entity model for the ACE05, SciERC, RadGraph datasets
`pipe1_ner_tokaux_attrcls_sent.py`: the entity model for the CXRGraph dataset (having an extra task of entity attribute classification)
`pipe2_re_tokaux_sent.py`: the relation model for all datasets (num_extra_sent_for_candi_ent=0)

We run main experiments with 8 different seeds [22-25, 32-35]. Other hyper-parameters are inherited from [PURE](https://github.com/princeton-nlp/PURE) and [PL-Marker](https://github.com/thunlp/PL-Marker?tab=readme-ov-file).

### Model Variants

`pipe1_ner_tokaux_doc.py` and `pipe1_ner_tokaux_attrcls_doc.py`: the entity model variants that takes an entire document as a data instance rather than a sentence
`pipe2_re_tokaux_sent.py` (num_extra_sent_for_candi_ent=n): the relation model variant that allows obtaining candidate entities from n addtitional sentences

## Inference

Our CXRGraph inference data are obtained via the scripts in `./inference`. The codes are obtained from `pipe1_ner_tokaux_attrcls_sent.py` and `pipe2_re_tokaux_sent.py` with minimal modification to do inference and output data.
