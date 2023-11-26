## Imagenet_v2 all_cls:
python3 -W ignore main.py --dataset imagenet_v2 --test_mode all_cls --templates_type clip_all --silent
python3 -W ignore main.py --dataset imagenet_v2 --test_mode all_cls --templates_type clip_photo --silent
python3 -W ignore main.py --dataset imagenet_v2 --test_mode all_cls --templates_type our --prompts_per_cls 10 --prompt_words_num 1 --silent
python3 -W ignore main.py --dataset imagenet_v2 --test_mode all_cls --templates_type our --prompts_per_cls 100 --prompt_words_num 1 --silent
python3 -W ignore main.py --dataset imagenet_v2 --test_mode all_cls --templates_type our --prompts_per_cls 1000 --prompt_words_num 1 --silent
python3 -W ignore main.py --dataset imagenet_v2 --test_mode all_cls --templates_type our --prompts_per_cls 1000 --prompt_words_num 2 --silent

## Imagenet_v2 my_cls_only:
# python3 -W ignore main.py --dataset imagenet_v2_100 --test_mode my_cls_only --templates_type clip_all --silent
# python3 -W ignore main.py --dataset imagenet_v2_100 --test_mode my_cls_only --templates_type clip_photo --silent
# python3 -W ignore main.py --dataset imagenet_v2_100 --test_mode my_cls_only --templates_type our --prompts_per_cls 10 --prompt_words_num 1 --silent

python3 -W ignore main.py --dataset imagenet_v2_100 --test_mode my_cls_only --templates_type our --prompts_per_cls 100 --prompt_words_num 1 --silent --gpt_prompts
python3 -W ignore main.py --dataset imagenet_v2_100 --test_mode my_cls_only --templates_type our --prompts_per_cls 100 --prompt_words_num 2 --silent --gpt_prompts
python3 -W ignore main.py --dataset imagenet_v2_100 --test_mode my_cls_only --templates_type our --prompts_per_cls 100 --prompt_words_num 3 --silent --gpt_prompts


# python3 -W ignore main.py --dataset imagenet_v2_100 --test_mode my_cls_only --templates_type our --prompts_per_cls 1000 --prompt_words_num 1 --silent
# python3 -W ignore main.py --dataset imagenet_v2_100 --test_mode my_cls_only --templates_type our --prompts_per_cls 1000 --prompt_words_num 2 --silent

## Imagenet_v2 my_cls_among_all:
# python3 -W ignore main.py --dataset imagenet_v2_100 --test_mode my_cls_among_all --templates_type clip_all --silent
# python3 -W ignore main.py --dataset imagenet_v2_100 --test_mode my_cls_among_all --templates_type clip_photo --silent
# python3 -W ignore main.py --dataset imagenet_v2_100 --test_mode my_cls_among_all --templates_type our --prompts_per_cls 1 --prompt_words_num 1 --silent
# python3 -W ignore main.py --dataset imagenet_v2_100 --test_mode my_cls_among_all --templates_type our --prompts_per_cls 10 --prompt_words_num 1 --silent

python3 -W ignore main.py --dataset imagenet_v2_100 --test_mode my_cls_among_all --templates_type our --prompts_per_cls 100 --prompt_words_num 1 --silent --gpt_prompts
python3 -W ignore main.py --dataset imagenet_v2_100 --test_mode my_cls_among_all --templates_type our --prompts_per_cls 100 --prompt_words_num 2 --silent --gpt_prompts
python3 -W ignore main.py --dataset imagenet_v2_100 --test_mode my_cls_among_all --templates_type our --prompts_per_cls 100 --prompt_words_num 3 --silent --gpt_prompts


# python3 -W ignore main.py --dataset imagenet_v2_100 --test_mode my_cls_among_all --templates_type our --prompts_per_cls 1000 --prompt_words_num 1 --silent
# python3 -W ignore main.py --dataset imagenet_v2_100 --test_mode my_cls_among_all --templates_type our --prompts_per_cls 1000 --prompt_words_num 2 --silent
# python3 -W ignore main.py --dataset imagenet_v2_100 --test_mode my_cls_among_all --templates_type our --prompts_per_cls 1000 --prompt_words_num 3 --silent
