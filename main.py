import torch
import torchvision

import numpy as np
import torch
import clip
from tqdm import tqdm
from pkg_resources import packaging

import argparse


# My:
from get_data import get_dataset_imagenet, get_dataset_imagenet_v2, get_dataset_cifar10
from old.utils import zeroshot_classifier, accuracy


def setup(args):

    ### Load model:

    model, transforms = clip.load(args.model_name)

    print("[INFO] Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("[INFO] Context length:", model.context_length, ", Vocab size:", model.vocab_size)


    ### Load data:


    if args.dataset == "imagenet":
        images, classes, templates = get_dataset_imagenet(transform=transforms, 
                                                            templates=args.templates_type,
                                                            num_classes='full')
        model.visual.input_resolution = 224

    elif args.dataset == "imagenet_100":
        images, classes, templates = get_dataset_imagenet(transform=transforms, 
                                                            templates=args.templates_type,
                                                            num_classes='100')
        model.visual.input_resolution = 224

    elif args.dataset == "imagenet_v2":
        images, classes, templates = get_dataset_imagenet_v2(transform=transforms, 
                                                            templates=args.templates_type,
                                                            num_classes='full')
        model.visual.input_resolution = 224

    elif args.dataset == "imagenet_v2_100":
        images, classes, templates = get_dataset_imagenet_v2(transform=transforms, 
                                                            templates=args.templates_type,
                                                            num_classes='100')
        model.visual.input_resolution = 224

    elif args.dataset == "cifar10":
        images, classes, templates = get_dataset_cifar10(transform=transforms, 
                                                            templates=args.templates_type)
        model.visual.input_resolution = 32 # 224 originally

    # elif args.dataset == "cifar100":
    #     num_classes=100
    #     dataset_path = ''

    #     images, classes, templates = get_dataset_cifar10(transform=transforms, 
    #                                                         templates=args.templates_type, location=dataset_path)
    #     model.visual.input_resolution = 32 # 224 originally

    else:
        raise Exception("[ERROR] Dataset name is not in the list") 

    test_loader = torch.utils.data.DataLoader(images, batch_size=32,
                                            shuffle=False, num_workers=8) #2)

    print(f"[INFO] {len(classes)} classes, {len(templates)} templates")
    print("[INFO] Input resolution:", model.visual.input_resolution)

    #print(templates)

    ### Prepare classification weights:

    zeroshot_weights, auxillary = zeroshot_classifier(model, classes, templates, 
                                                      test_mode=args.test_mode, aux=True)
    #print(auxillary)

    if args.test_mode != 'my_cls_among_all':
        if len(auxillary) != len(classes):
            raise Exception(f"[ERROR] Number of classes in prepared promts {len(auxillary)} \
                            is not the same as number of classes in the dataset {len(classes)}")

    print("[INFO] Classificator shape:", zeroshot_weights.shape)
    print("[INFO] Prompts per class:", len(auxillary[0]))

    return model, test_loader, zeroshot_weights


def test(args, model, test_loader, zeroshot_weights, mode='all_cls'):

    if mode == 'my_cls_among_all' or mode == 'my_cls_only' :
        from get_data import get_indices_imagenet100
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
                                              'imagenet', 'imagenet_100'], 
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


    #args = parser.parse_known_args()[0]
    args, unknown = parser.parse_known_args()                 
    # args = parser.parse_args()
    print(args)

    model, test_loader, zeroshot_weights = setup(args)

    accuracy = test(args, model, test_loader, zeroshot_weights, mode=args.test_mode)



if __name__ == "__main__":
    main()
