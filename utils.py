import torch
import clip
from tqdm import tqdm


def zeroshot_classifier(model, classnames, templates, test_mode="all_cls", aux=False):
    with torch.no_grad():
        zeroshot_weights = []
        if aux:
            auxillary = []
        
        if test_mode == 'my_cls_among_all':
            from get_data import get_classes_imagenet
            classnames = get_classes_imagenet()

        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            if aux:
                auxillary.append(texts)
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


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

