import torch
from Project.CLIP.CLIP import clip
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from Project.CLIP.CLIP.clip.simple_tokenizer import SimpleTokenizer as _tokenizer
from captum.attr import visualization
from PIL import Image
import os
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import statistics
import pandas as pd


def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            # auxillary.append(texts)
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def top_acc(loader, model, zeroshot_weights):
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda()
            target = target.cuda()

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
    
    return top1, top5

#Interpret function
def interpret(image, texts, model, device, start_layer, start_layer_text):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1:
      # calculate index of last layer
      start_layer = len(image_attn_blocks) - 1

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]


    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    if start_layer_text == -1:
      # calculate index of last layer
      start_layer_text = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text

    return text_relevance, image_relevance

#relevance function
def show_image_relevance(image_relevance, image, orig_image):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(orig_image);
    axs[0].axis('off');

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    axs[1].imshow(vis);
    axs[1].axis('off');

def show_heatmap_on_text(text, text_encoding, R_text):
  CLS_idx = text_encoding.argmax(dim=-1)
  R_text = R_text[CLS_idx, 1:CLS_idx]
  text_scores = R_text / R_text.sum()
  text_scores = text_scores.flatten()
  print(text_scores)
  text_tokens=_tokenizer.encode(text)
  text_tokens_decoded=[_tokenizer.decode([a]) for a in text_tokens]
  vis_data_records = [visualization.VisualizationDataRecord(text_scores,0,0,0,0,0,text_tokens_decoded,1)]
  visualization.visualize_text(vis_data_records)
  
def plot(image):
    print(type(image))
    if type(image) == torch.Tensor:
        numpy_image = image.numpy()
        # convert numpy array to 8-bit integer
        uint8_image = np.uint8(numpy_image)
        # create PIL Image object from 8-bit integer array
        pil_image = Image.fromarray(uint8_image)
        plt.imshow(pil_image)
    else:
        plt.imshow(image)
    plt.show()
    
    
def image_loader(folder_path):
    classes = os.listdir(folder_path)
    classes.sort()
    print(classes)
    
    image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_images = {}
    folder_path = 'clip_autoPrompt/Data/easy classes/Images/'
    for i in range(1, 13):
        folder_name = classes[i-1]+"/IN"
        # print(folder_name)
        folder_contents = os.listdir(os.path.join(folder_path, folder_name))
        print(f'Loading images from folder {folder_name}...')
        images = []
        for image_name in folder_contents:
            image_path = os.path.join(folder_path, folder_name, image_name)
            image = Image.open(image_path)
            # plot(image)
            if image.mode != 'RGB':
                image = ImageOps.colorize(image.convert('L'), (0, 0, 0), (255, 255, 255))
            image = image_transforms(image)
            # plot(image)
            images.append(image)
        tensor = torch.stack(images)
        classname = classes[i-1]
        all_images[classname.split()[-1]] = tensor
        print(f'Loaded {len(images)} images into tensor of shape {tensor.shape}')
        
    return all_images

def get_similarities_per_class(all_images_dict, prompts_df, model):
# all_images_dict: a dictionary where keys are the classnames and values are a tensors of images
# prompts_df: dataframe of a .tsv file from the dataset
# Returns a dictionary of 12 keys (representing classes) {"cucumber":{value}, ...},
#   where each corresponding value is itself a dictionary of {Similarity Score: "Prompt"}.

    # Define device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/32", device)


    # Group prompts by class
    prompts_by_class = prompts_df.groupby(0)
    # print(prompts_by_class.head(5))

    # Calculate similarity scores for each prompt in each class
    similarities_by_class = {}
    for class_name, class_prompts in prompts_by_class:
        print("\n\n\nClass Name: ", class_name)
        print("Number of prompts in this class: ", len(class_prompts[1]))
        # Preprocess image
        image = all_images_dict[class_name][0] #getting first image of a class
        image = (image.unsqueeze(0)).to(device)
        print(image.shape)
        class_similarities = {}
        #class_prompts[1] is giving the prompt, while 0 index will give the class name, since it's a dataframe
        for idp, prompt in enumerate(class_prompts[1]):
            #print(idp, prompt)
            try:
              prompt_tensor = clip.tokenize(prompt).to(device)
              #print("Pass")
            except:
              print("Skipping the prompt:", prompt)
              #print("Skip:", prompt)

            logits_per_image, logits_per_text = model(image, prompt_tensor)
            score = logits_per_image.item()

            #class_similarities[score] = prompt
            class_similarities[prompt] = score

        similarities_by_class[class_name] = class_similarities
        print(len(similarities_by_class[class_name]))

    return similarities_by_class

def get_Avg_similarities_per_class(all_images_dict, prompts_df, model):
# all_images_dict: a dictionary where keys are the classnames and values are a tensors of images
# prompts_df: dataframe of a .tsv file from the dataset
# Returns a dictionary of 12 keys (representing classes) {"cucumber":{value}, ...},
#   where each corresponding value is itself a dictionary of {Similarity Score: "Prompt"}.

    # Define device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/32", device)

    # Group prompts by class
    prompts_by_class = prompts_df.groupby(0)
    # print(prompts_by_class.head(5))

    # Calculate similarity scores for each prompt in each class
    similarities_by_class = {}
    for class_name, class_prompts in prompts_by_class:
        print("\n\n\nClass Name: ", class_name)
        print("Number of prompts in this class: ", len(class_prompts[1]))
        # Preprocess image
        image = all_images_dict[class_name][0] #getting first image of a class
        image = (image.unsqueeze(0)).to(device)
        print(image.shape)
        class_similarities = {}
        #class_prompts[1] is giving the prompt, while 0 index will give the class name, since it's a dataframe
        # for prompt in class_prompts[1]:
        #     try:
        #       prompt_tensor = clip.tokenize(prompt).to(device)
        #     except:
        #         print("Skipping the prompt:", prompt)

        # for classname in tqdm(classnames):
        zeroshot_weights = []
        texts = list(class_prompts[1]) #format with class
        print(texts)
        texts = clip.tokenize(texts).cuda() #tokenize
        class_embeddings = model.encode_text(texts) #embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda() #zeroshot_weights = average of all prompts in a class
        print(image.shape)
        print(zeroshot_weights.shape)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100. * image_features @ zeroshot_weights
        # logits_per_image, logits_per_text = model(image, zeroshot_weights)
        average_score = logits
        # class_similarities[score] = prompt
        similarities_by_class[class_name] = average_score
        print((similarities_by_class))

    return similarities_by_class


## 12 classes

def get_class_prompt_dict(prompts_df):
    # prompts_df: dataframe of a .tsv file from the dataset, with 2 columns: column name and respective prompts
    # Returns a dictionary of keys (representing classes) {"cucumber":[prompt1, prompt2, ..], ...},
    # where each corresponding value is a list of prompts.

    # Group prompts by class
    prompts_by_class = prompts_df.groupby(0)

    class_prompt_dict = {}
    for class_name, class_prompts in prompts_by_class:
        prompts = list(class_prompts[1])
        class_prompt_dict[class_name] = prompts

    return class_prompt_dict

## 100 classes:

def get_class_prompt_dict(prompts_df, imagenet_classes_100_our, imagenet_indices_100, imagenet_classes):
    # prompts_df: dataframe of a .tsv file from the dataset, with 2 columns: column name and respective prompts
    # Returns a dictionary of keys (representing classes) {"cucumber":[prompt1, prompt2, ..], ...},
    # where each corresponding value is a list of prompts.

    # Group prompts by class
    prompts_by_class = prompts_df.groupby(0)

    class_prompt_dict = {}
    for class_name, class_prompts in prompts_by_class:
        prompts = list(class_prompts[1])
        #print(class_name)
        #print(class_prompts)

        index_our = imagenet_classes_100_our.index(class_name)
        index_in = imagenet_indices_100[index_our]
        class_name_in = imagenet_classes[index_in]
        class_prompt_dict[class_name_in] = prompts

    return class_prompt_dict

def sort_dict_by_key(d):
    #sorted_dict = dict(sorted(d.items()))
    sorted_dict = dict(sorted(d.items(), key=lambda x: x[1]))
    return sorted_dict

def get_prompts(classnames, templates, auxillary):
    auxillary = {}
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            auxillary[classname] = texts

    return auxillary

def results(dictionary_df):
  df_keys = list(dictionary_df.keys())
  for df_class in df_keys:
    print("\nFor Class ", df_class, "-")
    d = dictionary_df[df_class]

    # keys = list(d.keys())
    # values = list(d.values())

    keys = list(d.keys())
    values = list(d.values())

    #print(keys)
    #print(values)

    # print("Minimum score:", min(keys), "with Prompt:", d[min(keys)])
    # print("Maximum score:", max(keys), "with Prompt:", d[max(keys)])
    # print("Mean score:", statistics.mean(keys))
    # print("Median score:", statistics.median(keys))

    position_min = values.index(min(values))
    position_max = values.index(max(values))

    print("Minimum score:", min(values), "with Prompt:", keys[position_min])
    print("Maximum score:", max(values), "with Prompt:", keys[position_max])
    print("Mean score:", statistics.mean(values))
    print("Median score:", statistics.median(values))
    
#Calculating average prompts
def zeroshot_prompt_weights1(classname, dictionary_df, model):
    zeroshot_weights = []

    with torch.no_grad():
        texts = list(dictionary_df[classname].values()) #format with class
        print(texts)
        texts = clip.tokenize(texts).cuda() #tokenize
        class_embeddings = model.encode_text(texts) #embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    print(zeroshot_weights.shape)
    return zeroshot_weights

#Calculating average prompts
def zeroshot_prompt_weights_photo(classnames, dictionary_df, templates, model):
    zeroshot_weights = []

    with torch.no_grad():
        for classname in tqdm(classnames):

            #texts = list(dictionary_df[classname].values()) #format with class
            texts = dictionary_df[classname] # format with class

            print(texts)

            texts = [template.format(text) for text in texts for template in templates] #format with class

            print(texts)
            texts = clip.tokenize(texts).cuda() #tokenize
            #print(texts[0])
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    print(zeroshot_weights.shape)
    return zeroshot_weights

#Calculating average prompts
def zeroshot_prompt_weights2(classnames, dictionary_df, model):
    zeroshot_weights = []

    with torch.no_grad():
        for classname in tqdm(classnames):

            #texts = list(dictionary_df[classname].values()) #format with class
            texts = dictionary_df[classname] # format with class

            print(texts)
            texts = clip.tokenize(texts).cuda() #tokenize
            #print(texts[0])
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    print(zeroshot_weights.shape)
    return zeroshot_weights

def load_dataframes(DIFFICULTY, PROMPTS_NUMBER, STRICT):
    if DIFFICULTY == 'easy':
        if PROMPTS_NUMBER == 100:
            df_prompts10words = pd.read_csv('clip_autoPrompt/Data/easy classes/Texts/imagenet_prompts10words_easy_sent100_rand1678710684.tsv', sep='\t', header=None)
            df_prompts5words = pd.read_csv('clip_autoPrompt/Data/easy classes/Texts/imagenet_prompts5words_easy_sent100_rand1678710690.tsv', sep='\t', header=None)
            df_prompts3words = pd.read_csv('clip_autoPrompt/Data/easy classes/Texts/imagenet_prompts3words_easy_sent100_rand1678710695.tsv', sep='\t', header=None)
            df_prompts1words = pd.read_csv('clip_autoPrompt/Data/easy classes/Texts/imagenet_prompts1words_easy_sent100_rand1678710776.tsv', sep='\t', header=None)
            df_promptsSent = pd.read_csv('clip_autoPrompt/Data/easy classes/Texts/imagenet_promptsSent_easy_sent100_rand1678710677.tsv', sep='\t', header=None)
        elif PROMPTS_NUMBER == 1000:
            df_prompts10words = pd.read_csv('/content/clip_autoPrompt/Data/easy classes/Texts/1000/imagenet_prompts10words_easy_sent1000_rand1682944282.tsv', sep='\t', header=None)
            df_prompts5words = pd.read_csv('/content/clip_autoPrompt/Data/easy classes/Texts/1000/imagenet_prompts5words_easy_sent1000_rand1679743120.tsv', sep='\t', header=None)
            df_prompts3words = pd.read_csv('/content/clip_autoPrompt/Data/easy classes/Texts/1000/imagenet_prompts3words_easy_sent1000_rand1679743412.tsv', sep='\t', header=None)
            df_prompts1words = pd.read_csv('/content/clip_autoPrompt/Data/easy classes/Texts/1000/imagenet_prompts1words_easy_sent1000_rand1679743098.tsv', sep='\t', header=None)
            df_promptsSent = pd.read_csv('/content/clip_autoPrompt/Data/easy classes/Texts/1000/imagenet_promptsSent_easy_sent1000_rand1679743129.tsv', sep='\t', header=None)

    elif DIFFICULTY == 'mid':
        if PROMPTS_NUMBER == 100:
            if STRICT:
                df_prompts1words = pd.read_csv('clip_autoPrompt/Data/mid_100classes/100/strict/imagenet_mid_sent100_1words_rand1683056007.tsv', sep='\t', header=None)
            else:
                df_prompts10words = pd.read_csv('clip_autoPrompt/Data/mid_100classes/100/non_strict/imagenet_mid_sent100_10words_rand1682978576.tsv', sep='\t', header=None)
                df_prompts5words = pd.read_csv('clip_autoPrompt/Data/mid_100classes/100/non_strict/imagenet_mid_sent100_5words_rand1682978569.tsv', sep='\t', header=None)
                df_prompts4words = pd.read_csv('clip_autoPrompt/Data/mid_100classes/100/non_strict/imagenet_mid_sent100_4words_rand1682978566.tsv', sep='\t', header=None)
                df_prompts3words = pd.read_csv('clip_autoPrompt/Data/mid_100classes/100/non_strict/imagenet_mid_sent100_3words_rand1682978562.tsv', sep='\t', header=None)
                df_prompts2words = pd.read_csv('clip_autoPrompt/Data/mid_100classes/100/non_strict/imagenet_mid_sent100_2words_rand1682978558.tsv', sep='\t', header=None)
                df_prompts1words = pd.read_csv('clip_autoPrompt/Data/mid_100classes/100/non_strict/imagenet_mid_sent100_1words_rand1682978554.tsv', sep='\t', header=None)
                df_promptsSent = pd.read_csv('clip_autoPrompt/Data/mid_100classes/100/non_strict/imagenet_mid_sent100_Sent_rand1682978580.tsv', sep='\t', header=None)
    return df_prompts10words, df_prompts5words, df_prompts3words, df_prompts1words, df_promptsSent