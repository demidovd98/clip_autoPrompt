import torch
import clip
from tqdm import tqdm


def zeroshot_classifier(model, classnames, templates, test_mode="all_cls", aux=False):
    with torch.no_grad():
        zeroshot_weights = []
        if aux:
            auxillary = []
        
        if test_mode == 'my_cls_among_all':
            from data.get_dataset import get_classes_imagenet
            classnames = get_classes_imagenet()

        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            if aux:
                auxillary.append(texts)

            #print(texts[0])

            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    
    if aux:
        return zeroshot_weights, auxillary
    else:
        return zeroshot_weights


def zeroshot_classifier_our(model, classnames, templates, test_mode="all_cls", aux=False, photo_of=True, a_photo_of_a=True):
    zeroshot_weights = []
    if aux:
        auxillary = []

    from data.get_dataset import get_templates_imagenet


    if test_mode == 'my_cls_among_all':
        from data.get_dataset import get_classes_imagenet
        classnames = get_classes_imagenet(num_classes=1000, multi_name=True, as_dict=False)
    
    if photo_of:
        from data.get_dataset import get_templates_basic

    with torch.no_grad():
        for classname in tqdm(classnames):

            try:
                #texts = list(dictionary_df[classname].values()) #format with class
                texts = templates[classname] # format with class
                #print(texts[0])

                if photo_of:
                    template_photo = get_templates_basic(a_photo_of_a=a_photo_of_a)
                    texts = [template.format(text) for text in texts for template in template_photo] #format with class
                print(texts[0])

            except:
                #texts = classname
                print("[WARNING] No prompts found for classname:", classname, ", use class name as a prompt instead.")

                texts = classname.split(",")[0] if classname else [classname]
                #print(texts)

                if photo_of:
                    template_photo = get_templates_basic(a_photo_of_a=a_photo_of_a)
                    #template_photo = get_templates_imagenet()

                    texts = [template.format(texts) for template in template_photo] #format with class

            if aux:
                auxillary.append(texts)

            #print(texts[0])

            texts = clip.tokenize(texts).cuda() #tokenize
            #print(texts[0])
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        print(zeroshot_weights.shape)
    
    if aux:
        return zeroshot_weights, auxillary
    else:
        return zeroshot_weights



def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

