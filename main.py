import torch
import torchvision

import numpy as np
import torch
import clip
from tqdm import tqdm
from pkg_resources import packaging

import argparse


# My:
from data.get_dataset import get_dataset #get_dataset_imagenet, get_dataset_imagenet_v2, get_dataset_cifar10, get_dataset_cifar100
from data.get_dataset import get_templates_basic, get_templates

from utils import zeroshot_classifier, zeroshot_classifier_our, accuracy
from data.get_prompts import get_prompts



def setup(args):

    ### Load model:

    model, transforms = clip.load(args.model_name)

    print("[INFO] Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("[INFO] Context length:", model.context_length, ", Vocab size:", model.vocab_size)


    ### Load data:

    # print(transforms)
    # Compose(
    #     Resize(size=224, interpolation=bicubic, max_size=None, antialias=warn)
    #     CenterCrop(size=(224, 224))
    #     <function _convert_image_to_rgb at 0x7f9bfa64db40>
    #     ToTensor()
    #     Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    # )

    images, classes = get_dataset(args.dataset, transform=transforms, 
                                    templates_type=args.templates_type)


    '''
    get_dataset_func = get_dataset(args.dataset)

    # images, classes, templates = get_dataset_func(args.dataset, num_classes=num_classes, 
    #                                                     transform=transforms, 
    #                                                     templates=args.templates_type,
    #                                                     num_classes=num_classes)

    if args.dataset == "imagenet_v1":
        num_classes=1000
        images, classes = get_dataset_func(transform=transforms, 
                                                            templates_type=args.templates_type,
                                                            num_classes=num_classes)
        # model.visual.input_resolution = 224

    elif args.dataset == "imagenet_v1_100":
        num_classes=100
        images, classes = get_dataset_func(transform=transforms, 
                                                            templates_type=args.templates_type,
                                                            num_classes=num_classes)
        # model.visual.input_resolution = 224

    elif args.dataset == "imagenet_v2":
        num_classes=1000
        images, classes = get_dataset_func(transform=transforms, 
                                                            templates_type=args.templates_type,
                                                            num_classes=num_classes)
        # model.visual.input_resolution = 224

    elif args.dataset == "imagenet_v2_100":
        num_classes=100
        images, classes = get_dataset_func(transform=transforms, 
                                                            templates_type=args.templates_type,
                                                            num_classes=num_classes)
        # model.visual.input_resolution = 224

    elif args.dataset == "cifar10":
        num_classes=10
        images, classes = get_dataset_func(transform=transforms, 
                                                            templates_type=args.templates_type,
                                                            num_classes=num_classes)
        # model.visual.input_resolution = 32 # 224 originally

    elif args.dataset == "cifar100":
        num_classes=100
        images, classes = get_dataset_func(transform=transforms, 
                                                            templates_type=args.templates_type,
                                                            num_classes=num_classes)
        # model.visual.input_resolution = 32 # 224 originally

    else:
        raise Exception("[ERROR] Dataset name is not in the list") 
    '''
            

    test_loader = torch.utils.data.DataLoader(images, batch_size=32,
                                            shuffle=False, num_workers=8) #2)

    print(f"[INFO] Classes number: {len(classes)} ") #, {len(templates)} templates")
    print("[INFO] Input resolution:", model.visual.input_resolution)



    ### Prepare  prompts and classification weights:

    if args.templates_type == "our":
        templates = get_prompts(file_name="", dataset_name=args.dataset, classes=classes, division=args.prompt_words_num, prompts_num=args.prompts_per_cls)
        zeroshot_weights, auxillary = zeroshot_classifier_our(model, classes, templates, 
                                                            test_mode=args.test_mode, aux=True, 
                                                            photo_of=True, a_photo_of_a=True)
        #templates = list(templates.values())
        templates_num = sum([len(value) for value in templates.values()])

    elif args.templates_type == "clip_all":
        templates = get_templates(args.dataset)
        zeroshot_weights, auxillary = zeroshot_classifier(model, classes, templates, 
                                                        test_mode=args.test_mode, aux=True)
        templates_num = len(templates)        
    elif args.templates_type == "clip_photo":
        templates = get_templates_basic()
        zeroshot_weights, auxillary = zeroshot_classifier(model, classes, templates, 
                                                        test_mode=args.test_mode, aux=True)
        templates_num = len(templates)

    #print(auxillary)
    #print(templates)

    if args.test_mode != 'my_cls_among_all':
        if len(auxillary) != len(classes):
            raise Exception(f"[ERROR] Number of classes in prepared promts {len(auxillary)} \
                            is not the same as number of classes in the dataset {len(classes)}")

    print(f"[INFO] Total prompts number: {templates_num}")
    print("[INFO] Prompts per class:", len(auxillary[0]))
    print("[INFO] Classificator shape:", zeroshot_weights.shape)

    return model, test_loader, zeroshot_weights



def test(args, model, test_loader, zeroshot_weights, mode='all_cls'):

    if mode == 'my_cls_among_all' or mode == 'my_cls_only' :
        from data.get_dataset import get_indices_imagenet100
        imagenet_indices = get_indices_imagenet100()

    if mode == 'my_cls_only' :
        # #map = {x.item(): i for i, x in enumerate(imagenet_indices_12)}
        labels_to_ids = {k: v for v, k in enumerate(imagenet_indices)}

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(test_loader)):

            if args.dataset == 'cifar10':
                for img in images:
                    #print(img.shape)
                    if len(img.shape) == 2:
                        #print(img.shape)
                        img = np.stack([img] * 3, 2)

            images = images.cuda()
            target = target.cuda()

            if mode == 'my_cls_among_all' or mode == 'my_cls_only' :
                mask = torch.zeros(target.size(), dtype=torch.bool).cuda()
                for index in imagenet_indices:
                    # print(index)
                    # print(target)
                    #result = torch.nonzero(target[:] == imagenet_indices_ours)
                    mask_temp = (target == index)
                    #mask += mask_temp
                    #print(mask_temp)
                    #print(mask)
                    mask = torch.add(mask, mask_temp)
                    #print(mask)

                target_new = torch.masked_select(target, mask)
                #images = torch.masked_select(target, mask)
                #print(target_new)

                if len(target_new) > 0:
                    #print(mask)
                    #print(target_new)
                    #print(images.size())

                    if mode == 'my_cls_among_all':
                        target = target_new
                    elif mode == 'my_cls_only':
                        target = torch.tensor([labels_to_ids[x.item()] for x in target_new]).cuda()

                    images = images[mask,:]

                    # # predict
                    # image_features = model.encode_image(images)
                    # image_features /= image_features.norm(dim=-1, keepdim=True)
                    # logits = 100. * image_features @ zeroshot_weights

                    # # measure accuracy
                    # acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                    # top1 += acc1
                    # top5 += acc5
                    # n += images.size(0)
                else:
                    continue 

            # else:
            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    print(f"[INFO] Top-1 accuracy: {top1:.2f}", f"Top-5 accuracy: {top5:.2f}")

    return top1



def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", #required=True,
                        default='name',
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=['cifar10', 'cifar100',
                                              'imagenet_v2', 'imagenet_v2_100',
                                              'imagenet_v1', 'imagenet_v1_100'], 
                        default='imagenet_v2',
                        help="Which downstream task.")
    
    # parser.add_argument("--model_type", choices=["clip"],
    #                     default="clip",
    #                     help="Which architecture to use.")
    
    parser.add_argument("--model_name", choices=['RN50', 'RN101', 'RN50x4',
                                                 'RN50x16', 'RN50x64',
                                                 'ViT-B/32', 'ViT-B/16',
                                                 'ViT-L/14', 'ViT-L/14@336px'],
                                                 #print(clip.available_models())
                        default='ViT-B/32',
                        help="Which specific model to use.")

    parser.add_argument("--test_mode", #required=True,
                        choices=['all_cls', 'my_cls_only', 'my_cls_among_all'], 
                        default='all_cls',
                        help="Which classiciation mode.")

    parser.add_argument("--templates_type", #required=True,
                        choices=['clip_all', 'clip_photo', 'our'], 
                        default='clip_all',
                        help="Which classiciation mode.")

    parser.add_argument("--prompts_per_cls", type=int, #required=True,
                        choices=[1, 10, 100, 1000], 
                        default=1,
                        help="Number of prompts per class")

    parser.add_argument("--prompt_words_num", type=int, #required=True,
                        choices=[1, 2, 3, 5], 
                        default=1,
                        help="Number of words to the left and to the right from the class name in the prompt")
    

    #args = parser.parse_known_args()[0]
    args, unknown = parser.parse_known_args()                 
    # args = parser.parse_args()
    print(args)


    model, test_loader, zeroshot_weights = setup(args)

    accuracy = test(args, model, test_loader, zeroshot_weights, mode=args.test_mode)



if __name__ == "__main__":
    main()
