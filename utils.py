import torch
import clip
from tqdm import tqdm
import re

import statistics



def zeroshot_classifier(args, model, classnames, templates, test_mode="all_cls", aux=False, refine_by_score=False, train_loader=None):
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

            if refine_by_score:

                from data.get_dataset import get_templates_basic, get_templates
                templates_clip = get_templates(args.dataset)
                templates_clip_photo = get_templates_basic(a_photo_of_a=True) #True)
                
                templates_temp = [template.format(classname) for template in templates_clip_photo] #format with class

                threshold = prompts_analysis(args, model, templates_temp, train_loader, classnames, classname, refine_by_score=False, threshold=None)
                threshold = 0
                texts_test = prompts_analysis(args, model, texts, train_loader, classnames, classname, refine_by_score=True, threshold=threshold)
            # else:
            #     prompts_analysis(args, model, texts, train_loader, classnames, classname)


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


def zeroshot_classifier_our(args, model, classnames, templates, test_mode="all_cls", aux=False, photo_of=True, a_photo_of_a=True, gpt_prompts=False, refine_by_score=False, train_loader=None):
    zeroshot_weights = []
    if aux:
        auxillary = []

    from data.get_dataset import get_templates_imagenet


    if test_mode == 'my_cls_among_all':
        from data.get_dataset import get_classes_imagenet
        classnames = get_classes_imagenet(num_classes=1000, multi_name=True, as_dict=False)
    
    if photo_of:
        from data.get_dataset import get_templates_basic


    # gpt_prompts = True #False
    # if gpt_prompts:
    #     from data.get_dataset import get_classes_imagenet
    #     classnames = get_classes_imagenet(num_classes=100, multi_name=True, as_dict=False)


    with torch.no_grad():
        for classname in tqdm(classnames):

            #print(type(classname))
            # if len(classname) > 1:
            #     classname = classname[0]

            classname_multi = classname

            #gpt_prompts = True #False
            if gpt_prompts:
                multi_name = classname.split(",")
                #print("BEFORE", classname)
                if len(multi_name) > 1:
                    classname = multi_name[0]
                #print(classname)

            try:
                #texts = list(dictionary_df[classname].values()) #format with class
                texts = templates[classname] # format with class
                #print(texts[0])

                texts_refined = []
                for line in texts:
                    # print("____")
                    # print(line)
                    line = re.sub(r"[\`\"\—\#\$\%\&\'\(\)\*\+\,\–\-\/\:\;\<\=\>\?\@\[\\\]\^\?\!\_\`\{\|\}\~\«\»]", " ", line) 
                    # for word in text.split():
                    #     ind = i.find('_') 
                    #     i_end = re.sub(r"[\`\"\—\#\$\%\&\'\(\)\*\+\,\–\-\/\:\;\<\=\>\?\@\[\\\]\^\?\!\_\`\{\|\}\~\«\»]", " ", i) 
                    #     if i[:ind] == i_end[:ind]: 
                    #         i_end = i[:ind].replace('...', '.') 
                    #         if i[:ind] == i_end[:ind]: 
                    #print(line)

                    texts_refined.append(line)

                #print(texts_refined[0])


                if photo_of:
                    template_photo = get_templates_basic(a_photo_of_a=a_photo_of_a)

                    #texts = [template.format(text) for text in texts for template in template_photo] #format with class
                    texts = [template.format(text) for text in texts_refined for template in template_photo] #format with class
                else:
                    texts = texts_refined
                #print(texts[0])

            except:
                #texts = classname
                print("[WARNING] No prompts found for classname:", classname, ", use class name as a prompt instead.")

                texts = classname.split(",")[0] if classname else [classname]
                #print(texts)

                if photo_of:
                    template_photo = get_templates_basic(a_photo_of_a=a_photo_of_a)
                    #template_photo = get_templates_imagenet()

            if aux:
                auxillary.append(texts)

            #print(texts[0])


            if refine_by_score:

                from data.get_dataset import get_templates_basic, get_templates
                templates_clip = get_templates(args.dataset)
                templates_clip_photo = get_templates_basic(a_photo_of_a=True) #True)
                
                # if args.dataset[0:8] == "imagenet":
                #     # classname_single = classname.split(",")[0]
                #     classname_single = classname.split(",")[0] if classname else [classname]

                # else:
                #     classname_single = classname
                classname_single = classname.split(",")[0] if classname else [classname]

                templates_temp = [template.format(classname_single) for template in templates_clip_photo] #format with class

                print("___1. Find threshold:")
                threshold = prompts_analysis(args, model, templates_temp, train_loader, classnames, classname_multi, refine_by_score=False, threshold=None)

                print("___2. Refine prompts with threshold:")
                texts = prompts_analysis(args, model, texts, train_loader, classnames, classname_multi, refine_by_score=True, threshold=threshold)

                print("___3. Analyse refined prompts:")
                _ = prompts_analysis(args, model, texts, train_loader, classnames, classname_multi, refine_by_score=False, threshold=None)
            # else:
            #     prompts_analysis(args, model, texts, train_loader, classnames, classname)



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



def prompts_analysis(args, model, texts, train_loader, classnames, classname, refine_by_score=False, threshold=None):

    #analysis = True

    #refine_by_score=True
    #if refine_by_score:

    #from data.get_dataset import get_classes_cifar10

    if refine_by_score:
        texts_refined_by_score = []
        min_scores = []
    # else:
    #     clip_statistics = []


    if train_loader is not None:

        scores_by_class = {}
        scores_by_prompt = {}

        classname_id = classnames.index(classname)

        for text in texts:

            scores = []
            #print(text)

            text_tokenised = clip.tokenize(text).cuda() #tokenize

            #text = text.unsqueeze(0).cuda()
            with torch.no_grad():
                #top1, top5, n = 0., 0., 0.
                
                img_processed = 0
                for i, (images, targets) in enumerate(train_loader): #tqdm(train_loader)):

                    if classname_id in targets:
                        #print(images.size())
                        #print(targets)
                        #print(targets.size())

                        #print(text_tokenised)
                        #print(text_tokenised.size())

                        for idx, img in enumerate(images):

                            # print(classnames)
                            # print(classname)

                            if targets[idx] == classname_id:
                                pass
                            else:
                                continue

                            #target = targets[idx].cuda()
                            img = img.unsqueeze(0).cuda()

                            img_processed += 1
                            # print("target:", target)
                            # print("class_id:", classname_id)


                            #print(img.size())
                            #print(target)
                            # if args.dataset == 'cifar10' or args.dataset == 'cifar100'):
                            #     for img in images:
                            #         #print(img.shape)
                            #         if len(img.shape) == 2:
                            #             #print(img.shape)
                            #             img = np.stack([img] * 3, 2)

                            logits_per_image, logits_per_text = model(img, text_tokenised)

                            score = logits_per_image.item()
                            #print(score)
                            
                            # if analysis:
                            scores.append(score)

                            ''' try score comparison per image (not only min/avg)?
                            if refine_by_score:
                                min_scores.append(min(scores))

                            else:
                                clip_statistics.append(min(scores))
                                #clip_statistics = min(scores)
                                #print(clip_statistics)
                            '''

                            if img_processed >= 10:
                                #print(img_processed)
                                # if analysis:
                                scores_by_prompt[text] = scores

                                #print(text)
                                #print(f"Min: {min(scores)}, Max: {max(scores)}, Mean: {statistics.mean(scores)}, Median: {statistics.median(scores)}")
                                
                                if not refine_by_score:
                                    #clip_statistics.append(min(scores))
                                    clip_statistics = min(scores)
                                    #print(clip_statistics)                            

                                break

                        if img_processed >= 10:
                            break   

                        #print(i*32)

            if refine_by_score:
                #print(min(scores), threshold)
                if min(scores) >= threshold: #[img_processed]: # try not (min >= min) but (avg/median >= min) ? # try score comparison per each image (not only min/avg)?
                    texts_refined_by_score.append(text)

            #print(len(scores_by_prompt[text]))

            #print(scores_by_prompt)

        #if analysis:
        scores_by_class[classname] = scores_by_prompt

        # stats_min = 0
        # stats_max = 0
        # stats_avg = 0
        # stats_med = 0

        items_all = []
        for scores_temp in scores_by_prompt.values():
            items_all.extend(scores_temp)

            # if stats_min = 0: stats_min= 100
            # if stats_max = 0: stats_max= 0
            # if stats_avg = 0: stats_avg= statistics.mean(scores_temp)
            # if stats_med = 0: statistics.median(scores_temp)=100

            # stats_min = min(min(scores_temp), stats_min)
            # stats_max = max(max(scores_temp), stats_max)
            # stats_avg = statistics.mean([statistics.mean(scores_temp), stats_avg])
            # stats_med = statistics.median([statistics.median(scores_temp), stats_med])

        print(f"Clas Min: {min(items_all)}, Clas Max: {max(items_all)}, Clas Mean: {statistics.mean(items_all)}, Clas Median: {statistics.median(items_all)}")


        #print(len(scores_by_class[classname]))

    print(scores_by_class[classname])

    
    if refine_by_score:
        print(texts_refined_by_score)
        print("len", len(texts), len(texts_refined_by_score))
        return texts_refined_by_score
    else:
        print(clip_statistics)
        return clip_statistics




def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

