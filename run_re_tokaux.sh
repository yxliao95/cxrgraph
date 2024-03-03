#!/bin/bash

for ((i=22; i<=25; i++)); do
    python /root/workspace/cxr_structured_report/pipe2_re_tokaux_sent.py --from_bash --output_name radgraph_pipe2_re_tokaux_sent0_seed_$i --seed $i --task radgraph --num_extra_sent_for_candi_ent 0
done

for ((i=32; i<=35; i++)); do
    python /root/workspace/cxr_structured_report/pipe2_re_tokaux_sent.py --from_bash --output_name radgraph_pipe2_re_tokaux_sent0_seed_$i --seed $i --task radgraph --num_extra_sent_for_candi_ent 0
done

python /root/workspace/cxr_structured_report/test_email.py

# nohup /root/workspace/cxr_structured_report/run_re_tokaux.sh > nohup.log 2>&1 &