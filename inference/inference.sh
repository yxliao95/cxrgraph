#!/bin/bash

python /root/workspace/cxr_structured_report/inference/inference_ent.py
python /root/workspace/cxr_structured_report/inference/inference_rel.py

python /root/workspace/cxr_structured_report/test_email.py

# nohup /root/workspace/cxr_structured_report/inference/inference.sh > nohup.log 2>&1 &