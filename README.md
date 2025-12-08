# ECSMT-Framework Mitigate Backdoor Attack on GNN

A PyTorch implementation of "GCleaner: Defending against Backdoor Attack on Graph Neural Networks" for the unlearning part/ main and we enhance/ expand the GCleaner framework by applying on it our ECSMT Framework that adds/ implements three mitigation techniques to improve its performance and decrease its computation costs. The implemented techniques are Robust training, graph regularization, and data sanitization. Our research project is titled
"
BRIDGING THE GAP BETWEEN NETWORK SCIENCE AND COMPLEX SYSTMS TO IDENTIFY AND MITIGATE CYBER RISK: IDENTIFY AND MITIGATE BACKDOOR ATTACKS ON GRAPH NEURAL NETWORK AND ON COMPLEX SYSTEM 
"

## Requirements

```
python==3.8.19
torch==2.2.2
torch_geometric==2.5.2
scikit-learn==1.3.0
```

## Quick Start

### Run in AIDS

```sh
python main.py --dataset AIDS --model GIN --split_ratio 0.8 --injection_ratio 0.05 --trigger_ratio 0.2 --target_label 1 --attack_method subgraph --epoch 50
```



