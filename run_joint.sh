#!/bin/bash

# RadGraph
python /root/workspace/cxr_structured_report/pipe1_ner_tokaux_sent.py --from_bash --bert_path /root/autodl-tmp/offline_models/bert-base-uncased --output_name pipe1_ner34_radgraph_bert
python /root/workspace/cxr_structured_report/pipe1_ner_tokaux_sent.py --from_bash --bert_path /root/autodl-tmp/offline_models/biobert-base-cased --output_name pipe1_ner34_radgraph_bio_bert
python /root/workspace/cxr_structured_report/pipe1_ner_tokaux_sent.py --from_bash --bert_path /root/autodl-tmp/offline_models/bio-clinical-bert-base-cased --output_name pipe1_ner34_radgraph_bio_clin_bert
python /root/workspace/cxr_structured_report/pipe1_ner_tokaux_sent.py --from_bash --bert_path /root/autodl-tmp/offline_models/biomedbert-base-uncased --output_name pipe1_ner34_radgraph_bio_med_bert
python /root/workspace/cxr_structured_report/pipe1_ner_tokaux_sent.py --from_bash --bert_path /root/autodl-tmp/offline_models/bluebert-pubmed-mimic-base-uncased --output_name pipe1_ner34_radgraph_blue_bert

python /root/workspace/cxr_structured_report/pipe2_re_tokaux_sent.py --from_bash --bert_path /root/autodl-tmp/cxr_structured_report/models/pipe1_ner34_radgraph_bert --data_path /root/autodl-tmp/cxr_structured_report/outputs/pipe1_ner34_radgraph_bert/pred_ner --output_name pipe2_re35_radgraph_bert --seed 35 --task radgraph
python /root/workspace/cxr_structured_report/pipe2_re_tokaux_sent.py --from_bash --bert_path /root/autodl-tmp/cxr_structured_report/models/pipe1_ner34_radgraph_bio_bert --data_path /root/autodl-tmp/cxr_structured_report/outputs/pipe1_ner34_radgraph_bio_bert/pred_ner --output_name pipe2_re35_radgraph_bio_bert --seed 35 --task radgraph
python /root/workspace/cxr_structured_report/pipe2_re_tokaux_sent.py --from_bash --bert_path /root/autodl-tmp/cxr_structured_report/models/pipe1_ner34_radgraph_bio_clin_bert --data_path /root/autodl-tmp/cxr_structured_report/outputs/pipe1_ner34_radgraph_bio_clin_bert/pred_ner --output_name pipe2_re35_radgraph_bio_clin_bert --seed 35 --task radgraph
python /root/workspace/cxr_structured_report/pipe2_re_tokaux_sent.py --from_bash --bert_path /root/autodl-tmp/cxr_structured_report/models/pipe1_ner34_radgraph_bio_med_bert --data_path /root/autodl-tmp/cxr_structured_report/outputs/pipe1_ner34_radgraph_bio_med_bert/pred_ner --output_name pipe2_re35_radgraph_bio_med_bert --seed 35 --task radgraph
python /root/workspace/cxr_structured_report/pipe2_re_tokaux_sent.py --from_bash --bert_path /root/autodl-tmp/cxr_structured_report/models/pipe1_ner34_radgraph_blue_bert --data_path /root/autodl-tmp/cxr_structured_report/outputs/pipe1_ner34_radgraph_blue_bert/pred_ner --output_name pipe2_re35_radgraph_blue_bert --seed 35 --task radgraph

# CXRGraph
python /root/workspace/cxr_structured_report/pipe1_ner_tokaux_attrcls_sent.py --from_bash --bert_path /root/autodl-tmp/offline_models/bert-base-uncased --output_name pipe1_ner35_cxrgraph_bert
python /root/workspace/cxr_structured_report/pipe1_ner_tokaux_attrcls_sent.py --from_bash --bert_path /root/autodl-tmp/offline_models/biobert-base-cased --output_name pipe1_ner35_cxrgraph_bio_bert
python /root/workspace/cxr_structured_report/pipe1_ner_tokaux_attrcls_sent.py --from_bash --bert_path /root/autodl-tmp/offline_models/bio-clinical-bert-base-cased --output_name pipe1_ner35_cxrgraph_bio_clin_bert
python /root/workspace/cxr_structured_report/pipe1_ner_tokaux_attrcls_sent.py --from_bash --bert_path /root/autodl-tmp/offline_models/biomedbert-base-uncased --output_name pipe1_ner35_cxrgraph_bio_med_bert
python /root/workspace/cxr_structured_report/pipe1_ner_tokaux_attrcls_sent.py --from_bash --bert_path /root/autodl-tmp/offline_models/bluebert-pubmed-mimic-base-uncased --output_name pipe1_ner35_cxrgraph_blue_bert


python /root/workspace/cxr_structured_report/pipe2_re_tokaux_sent.py --from_bash --bert_path /root/autodl-tmp/cxr_structured_report/models/pipe1_ner35_cxrgraph_bert --data_path /root/autodl-tmp/cxr_structured_report/outputs/pipe1_ner35_cxrgraph_bert/pred_ner --output_name pipe2_re23_cxrgraph_bert --seed 23 --task cxrgraph
python /root/workspace/cxr_structured_report/pipe2_re_tokaux_sent.py --from_bash --bert_path /root/autodl-tmp/cxr_structured_report/models/pipe1_ner35_cxrgraph_bio_bert --data_path /root/autodl-tmp/cxr_structured_report/outputs/pipe1_ner35_cxrgraph_bio_bert/pred_ner --output_name pipe2_re23_cxrgraph_bio_bert --seed 23 --task cxrgraph
python /root/workspace/cxr_structured_report/pipe2_re_tokaux_sent.py --from_bash --bert_path /root/autodl-tmp/cxr_structured_report/models/pipe1_ner35_cxrgraph_bio_clin_bert --data_path /root/autodl-tmp/cxr_structured_report/outputs/pipe1_ner35_cxrgraph_bio_clin_bert/pred_ner --output_name pipe2_re23_cxrgraph_bio_clin_bert --seed 23 --task cxrgraph
python /root/workspace/cxr_structured_report/pipe2_re_tokaux_sent.py --from_bash --bert_path /root/autodl-tmp/cxr_structured_report/models/pipe1_ner35_cxrgraph_bio_med_bert --data_path /root/autodl-tmp/cxr_structured_report/outputs/pipe1_ner35_cxrgraph_bio_med_bert/pred_ner --output_name pipe2_re23_cxrgraph_bio_med_bert --seed 23 --task cxrgraph
python /root/workspace/cxr_structured_report/pipe2_re_tokaux_sent.py --from_bash --bert_path /root/autodl-tmp/cxr_structured_report/models/pipe1_ner35_cxrgraph_blue_bert --data_path /root/autodl-tmp/cxr_structured_report/outputs/pipe1_ner35_cxrgraph_blue_bert/pred_ner --output_name pipe2_re23_cxrgraph_blue_bert --seed 23 --task cxrgraph


python /root/workspace/cxr_structured_report/test_email.py

# nohup /root/workspace/cxr_structured_report/run_joint.sh > nohup.log 2>&1 &