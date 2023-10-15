# Imagenet:
python3 new/main_new.py --dataset imagenet --test_mode all_cls --templates_type clip_all
python3 new/main_new.py --dataset imagenet --test_mode all_cls --templates_type clip_photo

python3 new/main_new.py --dataset imagenet_100 --test_mode my_cls_among_all --templates_type clip_all
python3 new/main_new.py --dataset imagenet_100 --test_mode my_cls_among_all --templates_type clip_photo

python3 new/main_new.py --dataset imagenet_100 --test_mode my_cls_only --templates_type clip_all
python3 new/main_new.py --dataset imagenet_100 --test_mode my_cls_only --templates_type clip_photo

# Imagenet_v2:
python3 new/main_new.py --dataset imagenet_v2 --test_mode all_cls --templates_type clip_all
python3 new/main_new.py --dataset imagenet_v2 --test_mode all_cls --templates_type clip_photo

python3 new/main_new.py --dataset imagenet_v2_100 --test_mode my_cls_among_all --templates_type clip_all
python3 new/main_new.py --dataset imagenet_v2_100 --test_mode my_cls_among_all --templates_type clip_photo

python3 new/main_new.py --dataset imagenet_v2_100 --test_mode my_cls_only --templates_type clip_all
python3 new/main_new.py --dataset imagenet_v2_100 --test_mode my_cls_only --templates_type clip_photo

# Cifar10:
python3 new/main_new.py --dataset cifar10 --test_mode all_cls --templates_type clip_all
python3 new/main_new.py --dataset cifar10 --test_mode all_cls --templates_type clip_photo