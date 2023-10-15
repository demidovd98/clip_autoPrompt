# clip_autoPrompt

Transferred the code to main.py and updated on github. Most of the repetitive codes are converted to functions in utils.py and ImagenetData_Class.py files. Probably you’ll have trouble running it. Please make sure to match the series of experiments you have to run with the initial notebook file.



```
source /apps/local/anaconda2023/conda_init.sh

conda activate clip_colab

python3 new/main_new.py --dataset imagenet --test_mode all_cls --templates_type clip_all
```

