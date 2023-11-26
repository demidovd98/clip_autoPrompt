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

    model.cuda()

    if not args.silent: print("[INFO] Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    if not args.silent: print("[INFO] Context length:", model.context_length, ", Vocab size:", model.vocab_size)


    ### Load data:

    if args.refine_by_score: # == "our_refined": # for cifar10 and cifar100
        images, classes, train_set = get_dataset(args.dataset, transform=transforms, 
                                        templates_type=args.templates_type,
                                        get_trainSet=True)
        if train_set is None: # temp solution
            train_set=images # use val dataset

    else:
        images, classes = get_dataset(args.dataset, transform=transforms, 
                                        templates_type=args.templates_type,
                                        get_trainSet=False)
            

    test_loader = torch.utils.data.DataLoader(images, batch_size=32,
                                            shuffle=False, num_workers=8) #2)

    if args.refine_by_score: # == "our_refined": # for cifar10 and cifar100
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                                shuffle=False, num_workers=8) #2)
    else:
        train_loader = None

    if not args.silent: print(f"[INFO] Classes number: {len(classes)} ") #, {len(templates)} templates")
    if not args.silent: print("[INFO] Input resolution:", model.visual.input_resolution)



    ### Prepare  prompts and classification weights:

    if args.templates_type == "our":
        templates = get_prompts(file_name="", dataset_name=args.dataset, classes=classes, division=args.prompt_words_num, prompts_num=args.prompts_per_cls, gpt_prompts=args.gpt_prompts)
        zeroshot_weights, auxillary = zeroshot_classifier_our(args, model, classes, templates, 
                                                            test_mode=args.test_mode, aux=True, 
                                                            photo_of=True, a_photo_of_a=True, #True)
                                                            gpt_prompts=args.gpt_prompts,
                                                            refine_by_score=args.refine_by_score,
                                                            train_loader=train_loader)
        #templates = list(templates.values())
        templates_num = sum([len(value) for value in templates.values()])

    elif args.templates_type == "clip_all":
        templates = get_templates(args.dataset)
        zeroshot_weights, auxillary = zeroshot_classifier(args, model, classes, templates, 
                                                        test_mode=args.test_mode, aux=True,
                                                        refine_by_score=args.refine_by_score,
                                                        train_loader=train_loader)
        templates_num = len(templates)        
    elif args.templates_type == "clip_photo":
        templates = get_templates_basic(a_photo_of_a=True) #True)
        zeroshot_weights, auxillary = zeroshot_classifier(args, model, classes, templates, 
                                                        test_mode=args.test_mode, aux=True,
                                                        refine_by_score=args.refine_by_score,
                                                        train_loader=train_loader)                                                        
        templates_num = len(templates)

    #print(auxillary)
    #print(templates)

    if args.test_mode != 'my_cls_among_all':
        if len(auxillary) != len(classes):
            raise Exception(f"[ERROR] Number of classes in prepared promts {len(auxillary)} \
                            is not the same as number of classes in the dataset {len(classes)}")

    if not args.silent: print(f"[INFO] Total prompts number: {templates_num}")
    if not args.silent: print("[INFO] Prompts per class:", len(auxillary[0]))
    if not args.silent: print("[INFO] Classificator shape:", zeroshot_weights.shape)

    return model, test_loader, zeroshot_weights



def test(args, model, test_loader, zeroshot_weights, mode='all_cls'):

    my_cls_with_all = False # False

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
            #print(target)

            if mode == ('my_cls_among_all') or (mode == 'my_cls_only') :
                mask = torch.zeros(target.size(), dtype=torch.bool).cuda()
                
                ### my with clip:
                if my_cls_with_all:
                    mask_reverse = torch.zeros(target.size(), dtype=torch.bool).cuda()
                ###

                for index in imagenet_indices:
                    # print(index)
                    # print(target)
                    #result = torch.nonzero(target[:] == imagenet_indices_ours)
                    mask_temp = (target == index)
                    
                    #print("mask", mask_temp)

                    ### my with clip:
                    if my_cls_with_all:
                        mask_temp_reverse = ~mask_temp #(target != index)
                        #print("mask rev:", mask_temp_reverse)

                    ###

                    #mask += mask_temp
                    #print(mask_temp)
                    #print(mask)
                    mask = torch.add(mask, mask_temp)
                    #print(mask)

                    ### my with clip:
                    if my_cls_with_all:
                        mask_reverse = ~mask #torch.add(mask_reverse, mask_temp_reverse)
                    #print(mask_reverse)
                    
                    ###

                    #print(mask)

                target_new = torch.masked_select(target, mask)

                ### my with clip:
                if my_cls_with_all:
                    target_new_reverse = torch.masked_select(target, mask_reverse)
                ###

                #images = torch.masked_select(target, mask)
                #print("new:", target_new)
                #print("new rev:", target_new_reverse)

                if len(target_new) > 0:
                    #print(mask)
                    #print(target_new)
                    #print(images.size())

                    if mode == 'my_cls_among_all':
                        target = target_new
                    elif mode == 'my_cls_only':
                        target = torch.tensor([labels_to_ids[x.item()] for x in target_new]).cuda()


                    ### my with clip:
                    if my_cls_with_all:
                        images_pos = images[mask,:]

                        # predict
                        image_features = model.encode_image(images_pos)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        logits = 100. * image_features @ zeroshot_weights

                        # measure accuracy
                        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                        top1 += acc1
                        top5 += acc5
                        n += images_pos.size(0)
                    ###
                    else:
                        images = images[mask,:]


                    ### my with clip:
                    if my_cls_with_all:
                        #mask_oppsoite = ~mask
                        #print(mask)
                        images_reverse = images[mask_reverse,:]
                        target_reverse = target_new_reverse
                    ###

                else:
                    ### my with clip:
                    if my_cls_with_all:
                        pass
                    ###
                    else:
                        continue

            # else:
            # predict
            if my_cls_with_all:
                if 0 < len(target_new) < 32:
                    image_features = model.encode_image(images_reverse)
                elif len(target_new) >= 32:
                    continue
                else:
                    image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ zeroshot_weights


            else:
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ zeroshot_weights

            # measure accuracy
            if my_cls_with_all:
                if len(target_new) > 0:
                    acc1, acc5 = accuracy(logits, target_reverse, topk=(1, 5))
                else:
                    acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            else:
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                
            top1 += acc1
            top5 += acc5

            if my_cls_with_all:
                if len(target_new) > 0:
                    n += images_reverse.size(0)
                else:
                    n += images.size(0)
            else:
                n += images.size(0)

    print("n_size:", n)
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
    
    parser.add_argument('--gpt_prompts', action='store_true',
                        help="Whether to use prompts from GPT")

    parser.add_argument('--silent', action='store_true',
                        help="Whether to print unnecessary details")


    parser.add_argument('--refine_by_score', action='store_true',
                        help="Whether to refine prompts with similarity score check")




    #args = parser.parse_known_args()[0]
    args, unknown = parser.parse_known_args()                 
    # args = parser.parse_args()
    print(args)


    model, test_loader, zeroshot_weights = setup(args)

    accuracy = test(args, model, test_loader, zeroshot_weights, mode=args.test_mode)



if __name__ == "__main__":
    main()
