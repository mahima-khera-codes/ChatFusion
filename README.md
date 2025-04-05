# LightGen: Efficient Image Generation through Knowledge Distillation and Direct Preference Optimization <br><sub>Official PyTorch Implementation</sub>

#### [<code>HF Checkpoint 🚀</code>](https://huggingface.co/Beckham808/LightGen) | [<code>Technical Report 📝</code>](https://arxiv.org/pdf/2503.08619)  


## 🦉 ToDo List

- [ ] DPO Post-proceesing Code Released 
- [ ] Release Complete Checkpoint.
- [ ] Add Accelerate Module.

## Env

```bash
conda create -n everlyn_video python=3.10
conda activate everlyn_video
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
# pip install -U xformers==0.0.26 --index-url https://download.pytorch.org/whl/cu121
pip install -r requierments.txt
```

## Prepare stage

```bash
huggingface-cli download --token hf_ur_token --resume-download stabilityai/stable-diffusion-3.5-large --local-dir stable-diffusion-3.5-large # Image VAE
huggingface-cli download --resume-download google/flan-t5-xxl --local-dir google/flan-t5-xxl # Text Encoder
huggingface-cli download --repo-type dataset --resume-download jackyhate/text-to-image-2M --local-dir text-to-image-2M # Dataset
```
untar script for text-to-image2M
```bash
#!/bin/bash

# Check if the 'untar' directory exists, and create it if it does not
mkdir -p untar

# Loop through all .tar files
for tar_file in *.tar; do
    # Extract the numeric part, for example 00001, 00002, ...
    dir_name=$(basename "$tar_file" .tar)
    
    # Create the corresponding directory
    mkdir -p "untar/$dir_name"
    
    # Extract the tar file to the corresponding directory
    tar -xvf "$tar_file" -C "untar/$dir_name"
    
    echo "Extraction completed: $tar_file to untar/$dir_name"
done

echo "All files have been extracted."
```
It may too large to cost much time to read this data in normal dataset, so we need to generate a json file first 
to accelerate this process, modify `scripts/generate_txt.py` then run it.

```bash
python generate_json.py
```

## Training
Script for the default setting, u can modify some setting in `scripts/run.sh`:
```bash
sh run.sh
```
<!-- `diffusion/__init__.py` maybe need reduce the time step -->

## Inference
Script for the default setting:
```bash
python pipeline_image.py
```

## ✨ Star History

[![Star History](https://api.star-history.com/svg?repos=XianfengWu01/LightGen)](https://star-history.com/#XianfengWu01/LightGen&Date)

## Cite
```
@article{wu2025lightgen,
  title={LightGen: Efficient Image Generation through Knowledge Distillation and Direct Preference Optimization},
  author={Wu, Xianfeng and Bai, Yajing and Zheng, Haoze and Chen, Harold Haodong and Liu, Yexin and Wang, Zihao and Ma, Xuran and Shu, Wen-Jie and Wu, Xianzu and Yang, Harry and others},
  journal={arXiv preprint arXiv:2503.08619},
  year={2025}
}
```