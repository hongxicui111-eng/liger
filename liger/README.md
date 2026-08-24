## [Unifying Generative and Dense Retrieval for Sequential Recommendation](https://arxiv.org/abs/2411.18814)

This repo contains the codebase for the paper: "Unifying Generative and Dense Retrieval for Sequential Recommendation".

<p align="center" width="100%">
    <img width="100%" src="figure.png">
</p>

### Setup
Please install and activate the environment through
```shell
conda env create -f env.yml
conda activate liger
pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

### Code Structure

The structure of this repository is as follows

    .     
    ├── configs                    # Contains all .yaml config files for Hydra to configure dataloading, train sequence generation, models, etc.
    │   ├── dataset                # general setup for amazon and steam dataset
    │   ├── hydra                  # slurm launcher configs
    │   ├── logging
    │   ├── method                 # baseline and our method setup
    │   └── main.yaml              # Main config file for Hydra
    ├── ID_generation              # Scrips for loading the dataset and train the semantic ID
    │   ├── ID                     # saved trained semantic ID for each item
    │   ├── preprocessing          # raw and processed dataset
    │   │   ├── processed
    │   │   ├── raw_data
    │   │   └── data_process.py    # preprocessing script
    │   ├── rqvae                  # RQ-VAE model definition
    │   ├── train_rqvae.py         # RQ-VAE training script
    │   └── utils.py               # utils for preprocessing
    ├── src                        # contains src for training and evaluating generative retrieval models
    ├── README.md
    ├── env.yaml
    ├── utils.py                   # utils for training
    └── run.py                     # Main entry point for training

### Experiment
To modify logging options, please edit the configuration file at [`configs/logging/wandb.yaml`](configs/logging/wandb.yaml). You can adjust settings such as the project name and logging behavior there.

For reproducing our method, refer to the provided scripts:
- [`scripts/ours.sh`](scripts/ours.sh): Main script for our method.
- [`scripts/tiger.sh`](scripts/tiger.sh): Script for the TIGER baseline.
- [`scripts/ablation.sh`](scripts/ablation.sh): Script for running our ablation studies.

When launching an experiment, the code will automatically:
1.	Download and preprocess the necessary datasets,
2.	Load the learned semantic IDs, or train the RQ-VAE model to generate semantic IDs,
3.	Start the main training process for generative retrieval.


#### Evaluation Metric Logging
The evaluation results are logged in Weights & Biases (wandb) by default. Each metric is named using the following pattern: `{retrieval_method}_{in/cold}_{val/test}/{Recall|NDCG}@k`:
* `{retrieval_method}`: Type of retrieval method used. 
    * `genret`: Generative retrieval.
    * `dense`: Dense retrieval.
    * `uni`: Our hybrid method that first retrieves top-$K$ candidates via generative retrieval, and then reranks the retrieved $K$ candidates together with the cold-start item using dense retrieval.
* `{in/cold}`: Evaluation subset.
    * `in`: In-set items.
	* `cold`: Cold-start items.
* `{val/test}`: Evaluation split.

Training-time evaluation is limited to either generative or dense retrieval to save time.

Test-time evaluation for our method includes varying the number of generative candidates $K$ before reranking. These are logged using the pattern: `uni_{in/cold}_test/GenK_{Recall|NDCG}@k`. Here, $K$ indicates the number of candidates retrieved by the generative module before reranking. In our paper, we report results with $K=20$ by default, but we also log results for $K = 40, 60, 80, 100$ for completeness.

### Citation
If you found our work useful, please consider citing it

    @article{yang2024unifying,
      title={Unifying Generative and Dense Retrieval for Sequential Recommendation},
      author={Yang, Liu and Paischer, Fabian and Hassani, Kaveh and Li, Jiacheng and Shao, Shuai and Li, Zhang Gabriel and He, Yun and Feng, Xue and Noorshams, Nima and Park, Sem and Long, Bo and Nowak, Robert and Gao, Xiaoli and Eghbalzadeh, Hamid},
      journal={arXiv preprint arXiv:2411.18814},
      year={2024}
    }


See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
The majority of "Unifying Generative and Dense Retrieval for Sequential Recommendation" is licensed under CC-BY-NC , as found in the [LICENSE](LICENSE) file., however portions of the project are available under separate license terms: rqvae is licensed Apache 2.0.

