#!/bin/bash

# for ((i=22; i<=25; i++)); do
#     python /root/workspace/cxr_structured_report/pipe1_ner_tokaux_sent.py --from_bash --output_name radgraph_pipe1_ner_tokaux_sent_seed_$i --seed $i --task radgraph
# done


# for ((i=32; i<=35; i++)); do
#     python /root/workspace/cxr_structured_report/pipe1_ner_tokaux_sent.py --from_bash --output_name radgraph_pipe1_ner_tokaux_sent_seed_$i --seed $i --task radgraph
# done

python /root/workspace/cxr_structured_report/pipe1_ner_tokaux_attrcls_doc.py
python /root/workspace/cxr_structured_report/pipe1_ner_tokaux_attrcls_sent.py

python /root/workspace/cxr_structured_report/test_email.py

# nohup /root/workspace/cxr_structured_report/run_ner_tokaux.sh > nohup.log 2>&1 &