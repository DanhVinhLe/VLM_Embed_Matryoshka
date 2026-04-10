# VLMEmbed
## Set up env
```bash
apt-get update
apt-get upgrade -y
cd VLM_Embed
python -m venv vlm
source vlm/bin/activate
```
## Set up
```
pip install -r requirements.txt
```
## Download dataset
1. Download the eval image file zip from huggingface (`optional`) 
```bash
cd VLM_Embed
wget https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip
unzip images.zip -d eval_images/
```
2. Download train image, it can take > 1 hour to download
```bash
cd VLM_Embed
bash download_traindata.sh
bash download_traindata_2.sh
```
3. Fix some line code 

Because of the error of code in **Transformers library**, run the following script to find the error and comment some lines: 

Just comment the following code, from line 140 to 143 in file **/vlm/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py**: 
```python
if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
    raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
else:
    size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
```
Or run `fix_lib.py` to fix: 
```python 
python fix_lib.py
```

## Training

Just run the scripts in folder `scripts`
- For run RKD: 
```bash
bash scripts/train_RKD.sh
bash scripts/train_distill_propose_V.sh
```

### Adaptive Matryoshka Stage-1: projection spec usage

For `--kd_loss_type adaptive_mrl_stage1`, you can explicitly control which projection matrices are trained with:

- `--stage1_projection_spec`: projection edges (`src_dim->dst_dim`)
- `--stage1_projection_weights`: optional per-edge loss weights

#### 1) Projection graph format

Use comma-separated pairs, each pair must be `larger_dim->smaller_dim` (or `larger_dim:smaller_dim`):

```bash
--stage1_projection_spec "2048->1024,2048->512,1024->512,1024->256,512->256,256->64"
```

This allows multiple larger dims projecting to the same smaller dim (for example both `2048->512` and `1024->512`).

If `--stage1_projection_spec ""` (empty), training defaults to **all valid larger->smaller pairs** from `--nested_dims` plus the model full dim.

#### 2) Per-edge weight format

Use comma-separated weighted edges:

```bash
--stage1_projection_weights "2048->1024:1.0,2048->512:0.7,1024->512:1.2,1024->256:0.9"
```

Any edge not listed gets default weight `1.0`.
Likewise, if an edge is not listed in `--orthogonal_pair_weights`, that edge uses orthogonal pair weight `1.0` and still uses the global `--orthogonal_weight`.

#### 3) Example command snippet

```bash
--kd_loss_type adaptive_mrl_stage1 \
--nested_dims 64 128 256 512 1024 2048 \
--stage1_phase all \
--stage1_projection_spec "2048->1024,2048->512,1024->512,1024->256,512->256,256->128,128->64" \
--stage1_projection_weights "2048->1024:1.0,2048->512:0.8,1024->512:1.0,1024->256:0.8,512->256:1.0,256->128:1.0,128->64:1.0" \
--orthogonal_weight 0.01
```

Orthogonality regularization is still applied per active projection edge.

## Inference & Evaluation
1. To evaluate our model on an MMEB dataset (e.g., MSCOCO_i2t), run:
```bash 
bash eval.sh
```

## Acknowledgement
- We have adapted code from [VLM2Vec]([https://github.com/TIGER-AI-Lab/VLM2Vec]) and [B3](https://github.com/raghavlite/B3)
