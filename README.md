# Hierarchical Mixture of Experts

### Publication

Weikai Li, Ding Wang, Zijian Ding, Atefeh Sohrabizadeh, Zongyue Qin, Jason Cong, Yizhou Sun. 



### Important files:

Since this is a very big project, we have a lot of files. However, if you want to make any adaption or modification, it is very likely that you only need to care about the following files:

- hls_sh: contains the shells that will be called and run during HLS
- src: contains all of our source codes
  - config.py: contains all the hyper-parameter settings
  - coreset.py: selects the data points by K-means for fine-tuning
  - data.py: data preprocessing and dataset split
  - main.py: the starting point of running code
  - model.py: contains the hierarchical MoE model
  - new_dse.py: runs the HLS
  - train.py: trains our model



### How to run our codes

##### 0, Set up the environment

Please install the packages in requirements.txt.

Due to the limited space, we cannot update our dataset into the supplementary material. Please download the "dse_database" directory from [HARP's repository](https://github.com/UCLA-VAST/HARP) and place it at "./dse_database".



##### 1, Pretrain on the source kernels

We first need to pre-train the model on the source kernels. Please set the following arguments in src/config.py:

- TASK: "regression" means regression model; "class" means classification model.

- MACHSUITE_KERNEL: set it as ["aes", "gemm-blocked", "gemm-ncubed", "spmv-crs", "spmv-ellpack", "stencil", "nw", "md", "stencil-3d"] to include the source kernels.

- poly_KERNEL: set it as ["2mm", "3mm", "adi", "atax", "bicg", "bicg-large", "covariance", "doitgen", "doitgen-red", "fdtd-2d", "gemm-p-large", "gemver", "gesummv", "heat-3d", "jacobi-1d", "mvt", "seidel-2d", "symm", "symm-opt", "syrk", "trmm", "mvt-medium", "atax-medium", "bicg-medium", "gesummv-medium", "symm-opt-medium"] to include the source kernels.

- transfer_learning: set it to False, since we are pre-training.

- moe_layers: please see the following options:

- | Option                                  | Explanation                                                  |
  | --------------------------------------- | ------------------------------------------------------------ |
  | []                                      | HARP                                                         |
  | ["gnn7"]                                | Single-level node MoE                                        |
  | ["pseudo_alone_w/o_gnn7"]               | Single-level block MoE                                       |
  | ["output_mlp"]                          | Single-level graph MoE                                       |
  | ["pseudo_alone", "gnn7"]                | Single-level node and block MoE                              |
  | ["gnn7", "output_mlp"]                  | Single-level node and graph MoE                              |
  | ["pseudo_alone_w/o_gnn7", "output_mlp"] | Single-level block and graph MoE                             |
  | ["pseudo_alone", "gnn7", "output_mlp"]  | Single-level MoE on all three granularities                  |
  | ["hierarchy-weighted-hidden"]           | Hierarchical MoE based on the second design of gating network |
  | ["hierarchy-weighted-input"]            | Hierarchical MoE based on the first design of gating network |

- moe_num_experts: the number of total experts (the default is 4).

- moe_k: the number of utilized experts (the default is 4).

- moe_lmbda: the weight of the low-level MoEs' regularization term $\alpha$  (the default is 5e-3).

- hierarchical_moe_lmbda: the weight of the high-level MoEs' regularization term $\beta$  (the default is 5e-3).

- hierarchical_moe_epoch: the number of warmup epochs in the two-stage training (the default is 500).

- hierarchical_alternate_train: whether to train the high-level expert models separately and jointly in turn during the second stage (the default is True).

- hierarchical_moe_component: the default is None. If you want to experiment with the hierarchical MoE model working on a single granularity, please set moe_layers to ["hierarchy-weighted-hidden"], and set hierarchical_moe_component to "gnn7", "pseudo_alone_w/o_gnn7", or "output_mlp" for node, block, or graph respectively.

- force_regen: If you are pre-training for the first time, please set it to True. It will automatically generate the dataset for you.  Otherwise please set it to False.

- train_mode: set it to "normal" (the default is "normal").

- load_data_to_device: set it to False if you want to save memory (the default is True). Disabling it can save the memory from about 30 GB to about 15 GB.

Then, you can use `python main.py` to pre-train. After pre-training, please look at the output and remember the model saving path. We use the validation set for early-stopping, so we should look for the model path that ends with "val_model_state_dict.pth".

If you want to use MAML, please set train_mode to "maml". If you want to run ProgSG, please refer to its [official repository](https://github.com/ZongyueQin/ProgSG).



##### 2, Fine-tune on the target kernels

After pre-training, we need to fine-tune the model on the target kernels. Please set the following arguments in src/config.py:

- TASK: "regression" means regression model; "class" means classification model.
- MACHSUITE_KERNEL: set it to [].
- poly_KERNEL: set it as ["fdtd-2d-large", "gemver-medium", "syr2k", "gemm-p", "jacobi-2d", "correlation", "trmm-opt"] to include the target kernels.
- transfer_learning: set it to True, since we are fine-tuning.
- model_path: set it to the pre-trained model path.
- force_regen: If you are fine-tuning the code for the first time, please set it to True. It will automatically generate the dataset for you. Otherwise please set it to False.
- coreset: set it to "kmeans" to use K-means to select the 50 data points per kernel for fine-tuning; set it to "random" to select the fine-tuning data points by the random split.
- train_mode: set it to "normal" (the default is "normal").

Then, you can use `python main.py` to fine-tune. After fine-tuning, please look at the output and remember the model saving path. During fine-tuning, we mimic the situation where we only have 50 data points per kernel,  so there is no validation set, so we should look for the model path that ends with "train_model_state_dict.pth" instead of "test_model_state_dict.pth".



##### 3, Run HLS

1. Please first download the AMD/Xilinx HLS tool, Vitis 2021.1, and also download the Merlin tool.
2. Running HLS requires special machines and environments, which most of normal servers do not support. If your machines are able to set up the environment, the second step is to modify the environment paths in the "hls_sh/run.sh" and "hls_sh/vitis_env.sh".
3. Then, you can use "src/run_new_dse.sh" to run HLS. Please specify the "model_path" (fine-tuned regression model path), "class_model_path" (fine-tuned classification model path), and "merlin_path" in the script.



##### 4, Run the expert assignment analysis

Please keep all the settings in src/config.py the same as fine-tuning, except for changing "train_mode" to "observe_moe_distribution", and setting "observe_moe_layer" to your desired MoE model.
