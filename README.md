# MT-Final-Project
```bash
nohup python -u att_ml.py --dynet-mem 3000 --dynet-gpu-ids 2 -src cs -tgt de > logs/cs_de.out 2>&1 &
nohup python -u att_ml.py --dynet-mem 3000 --dynet-gpu-ids 2 -src cs -tgt fr > logs/cs_fr.out 2>&1 &
nohup python -u att_ml.py --dynet-mem 3000 --dynet-gpu-ids 2 -src de -tgt cs > logs/de_cs.out 2>&1 &
nohup python -u att_ml.py --dynet-mem 3000 --dynet-gpu-ids 3 -src fr -tgt cs > logs/fr_cs.out 2>&1 &
nohup python -u att_ml.py --dynet-mem 3000 --dynet-gpu-ids 3 -src de -tgt fr > logs/de_fr.out 2>&1 &
nohup python -u att_ml.py --dynet-mem 3000 --dynet-gpu-ids 3 -src fr -tgt de > logs/fr_de.out 2>&1 &
```