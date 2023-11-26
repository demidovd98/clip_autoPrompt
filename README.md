# clip_autoPrompt


## Setup environment:
```bash
cd /l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt

source /apps/local/anaconda2023/conda_init.sh
conda activate clip_colab
#or
source /home/dmitry.demidov/anaconda3/bin/activate clip_colab
```

## Run:
```bash
# Classify with all CLIP prompts:
python3 -W ignore main.py --dataset cifar10 --test_mode all_cls --templates_type clip_all --silent

# Classify with "a photo of {class}" prompts:
python3 -W ignore main.py --dataset cifar10 --test_mode all_cls --templates_type clip_photo --silent

# Classify with our Search prompts:
python3 -W ignore main.py --dataset cifar10 --test_mode all_cls --templates_type our --prompts_per_cls 1000 --prompt_words_num 1 --silent

# Classify with our GPT prompts:
python3 -W ignore main.py --dataset cifar10 --test_mode all_cls --templates_type our --prompts_per_cls 1000 --prompt_words_num 1 --silent --gpt_prompts


# Search or preapre prompts:
python3 data/prompts.py
```



