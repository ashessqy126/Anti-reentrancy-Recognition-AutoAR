# Anti-reentrancy-Recognition-AutoAR
An unsupervised learning method that can reconigize smart contracts that are protected by anti-reentrancy patterns

**1 Install Requirements**

pip install pytorch, numpy, pytorch-geometric, slither-analyzer, Genism, scikit-learn

**2 Performance Evaluation**

python sys_evaluation.py

The result will show **recall, precision, false positive ratio (FPR), and false negative ratio (FNR)** for anti-reentrancy recognition on testing dataset


**3 Instructions for Running**

**3.1 Extracting Anti-reentrancy-related Graphs and Inital Embedding**

python antireentrancy_graph_embed.py [dataset_name] [target_graphs_dir]


**3.2 Recognizing whether graphs employ anti-reentrancy patterns**

python sys_recongize.py [source_graph_dir] [final_embedding_dir]

