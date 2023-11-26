## cifar10 all_cls:
python3 -W ignore main.py --dataset cifar10 --test_mode all_cls --templates_type clip_all --silent
python3 -W ignore main.py --dataset cifar10 --test_mode all_cls --templates_type clip_photo --silent

python3 -W ignore main.py --dataset cifar10 --test_mode all_cls --templates_type our --prompts_per_cls 10 --prompt_words_num 1 --silent
python3 -W ignore main.py --dataset cifar10 --test_mode all_cls --templates_type our --prompts_per_cls 100 --prompt_words_num 1 --silent
python3 -W ignore main.py --dataset cifar10 --test_mode all_cls --templates_type our --prompts_per_cls 1000 --prompt_words_num 1 --silent
python3 -W ignore main.py --dataset cifar10 --test_mode all_cls --templates_type our --prompts_per_cls 1000 --prompt_words_num 2 --silent


python3 -W ignore main.py --dataset cifar10 --test_mode all_cls --templates_type our --prompts_per_cls 100 --prompt_words_num 1 --silent --gpt_prompts
python3 -W ignore main.py --dataset cifar10 --test_mode all_cls --templates_type our --prompts_per_cls 100 --prompt_words_num 2 --silent --gpt_prompts
python3 -W ignore main.py --dataset cifar10 --test_mode all_cls --templates_type our --prompts_per_cls 100 --prompt_words_num 3 --silent --gpt_prompts
