# clip_autoPrompt

Transferred the code to main.py and updated on github. Most of the repetitive codes are converted to functions in utils.py and ImagenetData_Class.py files. Probably you’ll have trouble running it. Please make sure to match the series of experiments you have to run with the initial notebook file.



```bash
cd /l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt

source /apps/local/anaconda2023/conda_init.sh
conda activate clip_colab
#or
source /home/dmitry.demidov/anaconda3/bin/activate clip_colab

python3 -W ignore main.py --dataset cifar10 --test_mode all_cls --templates_type our --prompts_per_cls 100 --prompt_words_num 1 --silent
#or
python3 data/prompts.py # search for prompts
```



