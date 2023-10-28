import pandas as pd
import os


def get_prompts(file_name, dataset_name, classes, division, prompts_num): #, strict=True, options=True):

    root = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/"

    file_name = file_name
    # folder = folder #"/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/" + str(DATASET) + "_" + str(RAND)
    # folder_prompts = folder + "/prompts"
    # RAND = rand # 100_cifar: 1684427807, 10_cifar: 1684427942 ; 1000_100c_strict10_new: 1683237191, 100_100c_strict10_new: 1683237344, # 1000_100c_strict: 1682947137, # 100_100c_strict_new: 1683038132, # 100_100c_strict: 1682986352, # 100_100c: 1682891271, # 10_100c_strict: 1683038085 # 10_100c: 1682986245 # 100_12c: 1678543337 # 1000_12c: 1678705786

    path = os.path.join(root, file_name)

    ## CIFAR10:
    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/cifar10_cls10_p1_1698265068/prompts/prompts_full_sent1_1words_rand1698265066.tsv"

    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/cifar10_cls10_p10_1698265107/prompts/prompts_full_sent10_1words_rand1698265105.tsv"
    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/cifar10_cls10_p10_1698265107/prompts/prompts_full_sent10_2words_rand1698265105.tsv"

    #path =  "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/cifar10_cls10_p100_1698265134/prompts/prompts_full_sent100_1words_rand1698265132.tsv"
    #path =  "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/cifar10_cls10_p100_1698265134/prompts/prompts_full_sent100_2words_rand1698265132.tsv"

    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/cifar10_cls10_p1000_1698265205/prompts/prompts_full_sent1000_1words_rand1698265203.tsv"
    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/cifar10_cls10_p1000_1698265205/prompts/prompts_full_sent1000_2words_rand1698265203.tsv"


    ## CIFAR100:
    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/cifar100_cls100_1697655609 (1p)/prompts/prompts_full_sent1_1words_rand1697655609.tsv"

    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/cifar100_cls100_1697659502 (10p)/prompts/prompts_full_sent10_1words_rand1697659502.tsv"

    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/cifar100_cls100_p100_1697661000 (100p)/prompts/prompts_full_sent100_1words_rand1697661000.tsv"
    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/cifar100_cls100_p100_1697661000 (100p)/prompts/prompts_full_sent100_2words_rand1697661000.tsv"

    path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/cifar100_cls100_p1000_1697743745 (1000p)/prompts/prompts_full_sent1000_1words_rand1697743745.tsv"
    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/cifar100_cls100_p1000_1697743745 (1000p)/prompts/prompts_full_sent1000_2words_rand1697743745.tsv"


    ## ImageNet 100:
    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/imagenet100_1697538250 (1p)/prompts/prompts_mid_sent1_1words_rand1697538250.tsv"

    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/imagenet100_1697537792 (10p)/prompts/prompts_mid_sent10_1words_rand1697537792.tsv"

    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/imagenet100_1697476227 (100p)/prompts/prompts_mid_sent100_1words_rand1697476227.tsv"
    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/imagenet100_1697476227 (100p)/prompts/prompts_mid_sent100_2words_rand1697476227.tsv"

    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/imagenet100_1697491561 (1000p)/prompts/prompts_mid_sent1000_1words_rand1697491561.tsv"
    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/imagenet100_1697491561 (1000p)/prompts/prompts_mid_sent1000_2words_rand1697491561.tsv"


    ## ImageNet 1000:
    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/imagenet_cls1000_1697543234 (1p)/prompts/prompts_full_sent1_1words_rand1697642394.tsv"
    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/imagenet_cls1000_1697543234 (1p)/prompts/prompts_full_sent1_2words_rand1697642394.tsv"

    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/imagenet_cls1000_1697549557 (10p)/prompts/prompts_full_sent10_1words_rand1697642268.tsv"
    #path = "/l/users/20020067/Activities/CLIP_prompts/clip_autoPrompt/data/prompts_our/_good/imagenet_cls1000_1697549557 (10p)/prompts/prompts_full_sent10_2words_rand1697642268.tsv"


    # if not os.path.exists(path):


    dataset_name = dataset_name
    #DATASET = dataset # ['cifar10', 'imagenet']

    classes = classes
    #classes_temp = classes

    class_num = len(classes)
    #DIFFICULTY = 'mid' # 'easy', 'mid', 'full'
    #DIFFICULTY = difficulty # 'easy', 'mid', 'full'

    prompts_num = prompts_num
    #PROMPTS_NUMBER = 100 # 1000, 100
    #NUM_OF_SENTENCES = numb_of_sent # 1, 10, 100, 1000

    '''
    strict = strict
    #STRICT = True
    # STRICT = strict #True
    # STRICT_SYMB = strict_symb # optimal: 10, min: 7 (tractor), 6 (chihuahuan desert)

    options = options # True
    #OPTIONS = options # True
    '''

    division = division
    #DIVISION = division # 'Sent', '7words', '6words', '5words', '4words', '3words', '2words', '1words'

    

    ## Load prompts:

    # if DIFFICULTY == 'easy':
    # if PROMPTS_NUMBER == 100:
        # df_prompts10words = pd.read_csv('/content/clip_autoPrompt/Data/mid_100classes/100/non_strict/imagenet_mid_sent100_10words_rand1682978576.tsv', sep='\t',header = None)
        # df_prompts5words = pd.read_csv('/content/clip_autoPrompt/Data/mid_100classes/100/non_strict/imagenet_mid_sent100_5words_rand1682978569.tsv', sep='\t',header = None)
        # df_prompts4words = pd.read_csv('/content/clip_autoPrompt/Data/mid_100classes/100/non_strict/imagenet_mid_sent100_4words_rand1682978566.tsv', sep='\t',header = None)
        # df_prompts3words = pd.read_csv('/content/clip_autoPrompt/Data/mid_100classes/100/non_strict/imagenet_mid_sent100_3words_rand1682978562.tsv', sep='\t',header = None)
        # df_prompts2words = pd.read_csv('/content/clip_autoPrompt/Data/mid_100classes/100/non_strict/imagenet_mid_sent100_2words_rand1682978558.tsv', sep='\t',header = None)
        # df_prompts1words = pd.read_csv('/content/clip_autoPrompt/Data/mid_100classes/100/non_strict/imagenet_mid_sent100_1words_rand1682978554.tsv', sep='\t',header = None)
        # df_promptsSent = pd.read_csv('/content/clip_autoPrompt/Data/mid_100classes/100/non_strict/imagenet_mid_sent100_Sent_rand1682978580.tsv', sep='\t',header = None)
    
    prompts_df = pd.read_csv(path, sep='\t',header = None)

    print(prompts_df.shape)
    print(prompts_df)

    prompts_dict = get_class_prompt_dict(prompts_df, dataset_name, class_num)

    return prompts_dict




def get_class_prompt_dict(prompts_df, dataset_name, class_num):
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

        '''
        if (dataset_name[0:8] == "imagenet") and (class_num<1000):
            from data.get_dataset import get_classes_imagenet, get_indices_imagenet100
            classnames100 = get_classes_imagenet(num_classes=100, multi_name=True, as_dict=False)
            classnames1000 = get_classes_imagenet(num_classes=1000, multi_name=True, as_dict=False)

            index_our = classnames100.index(class_name)
            index_in = get_indices_imagenet100()[index_our]
            class_name_in = classnames1000[index_in]
            print(class_name_in)

            class_prompt_dict[class_name_in] = prompts
        else:
            class_prompt_dict[class_name] = prompts
        '''

        class_prompt_dict[class_name] = prompts


    return class_prompt_dict

