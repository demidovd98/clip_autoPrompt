import pandas as pd
from tqdm import tqdm

import re

import time
epoch_time = int(time.time())


DATASET = 'cifar10' # 'imagenet'
NUM_OF_SENTENCES = 10 # 1, 10, 100, 1000
DIFFICULTY = 'mid' # 'easy', 'mid', 'full'
DIVISION = '1words' # 'Sent', '7words', '6words', '5words', '4words', '3words', '2words', '1words'
RAND = 1684427942 # 100_cifar: 1684427807, 10_cifar: 1684427942 ; 1000_100c_strict10_new: 1683237191, 100_100c_strict10_new: 1683237344, # 1000_100c_strict: 1682947137, # 100_100c_strict_new: 1683038132, # 100_100c_strict: 1682986352, # 100_100c: 1682891271, # 10_100c_strict: 1683038085 # 10_100c: 1682986245 # 100_12c: 1678543337 # 1000_12c: 1678705786
OLD = False # False with b'text' for 12 classes
STRICT = True
STRICT_SYMB = 10 # optimal: 10, min: 7 (tractor), 6 (chihuahuan desert)
OPTIONS = True # True

ONE_CLASS = False # True, but False is better 
# NEED FIX: when separating all classes to single words then same words may be in the foreugnlist (sea lion , lion) 

'''
classes_easy = [ 'cucumber', 'mushroom', 'banana', 'pizza', 'bucket', 'umbrella', 'mailbox', 'microwave', 'rifle', 'torch', 'volleyball', 'taxi' ] # 12 w
#classes_easy = ["cucumber", "mushroom", "banana", "pizza", "bucket", "umbrella", "mailbox", "zebra", "rifle", "torch", "volleyball", "tractor"] # 12 w
#'lemon', 'hammer', 'tractor', 'zebra'

classes_mid = [ 'cricket', 'hamster', 'gazelle', 'gorilla', 'barometer', 'canoe', 'forklift', 'printer', 'submarine', 'ballplayer', 'volcano', 'mosque' ] # 12 w
#'golfcart', 'pineapple', 'violin', 'espresso'
'''

#classes_mid_100_list = ['goldfish, Carassius auratus', 'hammerhead, hammerhead shark', 'cock', 'hen', 'bald eagle, American eagle, Haliaeetus leucocephalus', 'scorpion', 'garden spider, Aranea diademata', 'black widow, Latrodectus mactans', 'tarantula', 'wolf spider, hunting spider', 'hummingbird', 'goose', 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 'wombat', 'jellyfish', 'flamingo', 'pelican', 'albatross, mollymawk', 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus', 'sea lion', 'Chihuahua', 'toy terrier', 'Scotch terrier, Scottish terrier, Scottie', 'German shepherd, German shepherd dog, German police dog, alsatian', 'Doberman, Doberman pinscher', 'Siberian husky', 'coyote, prairie wolf, brush wolf, Canis latrans', 'red fox, Vulpes vulpes', 'lion, king of beasts, Panthera leo', 'tiger, Panthera tigris', 'cheetah, chetah, Acinonyx jubatus', 'zebra', 'bison', 'gorilla, Gorilla gorilla', 'chimpanzee, chimp, Pan troglodytes', 'assault rifle, assault gun', 'backpack, back pack, knapsack, packsack, rucksack, haversack', 'cannon', 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM', 'castle', 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'church, church building', 'cinema, movie theater, movie theatre, movie house, picture palace', 'desktop computer', 'dishwasher, dish washer, dishwashing machine', 'forklift', 'fountain', 'iPod', 'parachute, chute', 'pickup, pickup truck', 'pillow', 'refrigerator, icebox', 'remote control, remote', 'restaurant, eating house, eating place, eatery', 'school bus', 'scoreboard', 'screen, CRT screen', 'shopping cart', 'stove', 'sunglasses, dark glasses, shades', 'syringe', 'table lamp', 'tank, army tank, armored combat vehicle, armoured combat vehicle', 'teapot', 'teddy, teddy bear', 'television, television system', 'toaster', 'toilet seat', 'torch', 'tractor', 'umbrella', 'vacuum, vacuum cleaner', 'vending machine', 'volleyball', 'wallet, billfold, notecase, pocketbook', 'wardrobe, closet, press', 'water bottle', 'water tower', 'web site, website, internet site, site', 'comic book', 'street sign', 'traffic light, traffic signal, stoplight', 'ice cream, icecream', 'cheeseburger', 'hotdog, hot dog, red hot', 'broccoli', 'cucumber, cuke', 'mushroom', 'strawberry', 'orange', 'lemon', 'pineapple, ananas', 'banana', 'pizza, pizza pie', 'burrito', 'red wine', 'espresso', 'seashore, coast, seacoast, sea-coast', 'volcano', 'ballplayer, baseball player']
classes_mid_100_list = ['goldfish, Carassius auratus', 'hammerhead, hammerhead shark', 'cock', 'hen', 'bald eagle, American eagle, Haliaeetus leucocephalus', 'scorpion', 'black widow, Latrodectus mactans', 'tarantula', 'wolf spider, hunting spider', 'hummingbird', 'goose', 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 'wombat', 'jellyfish', 'flamingo', 'pelican', 'albatross, mollymawk', 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus', 'sea lion', 'Chihuahua', 'Scotch terrier, Scottish terrier, Scottie', 'German shepherd, German shepherd dog, German police dog, alsatian', 'Doberman, Doberman pinscher', 'coyote, prairie wolf, brush wolf, Canis latrans', 'red fox, Vulpes vulpes', 'lion, king of beasts, Panthera leo', 'tiger, Panthera tigris', 'cheetah, chetah, Acinonyx jubatus', 'zebra', 'bison', 'gorilla, Gorilla gorilla', 'chimpanzee, chimp, Pan troglodytes', 'assault rifle, assault gun', 'backpack, back pack, knapsack, packsack, rucksack, haversack', 'cannon', 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM', 'castle', 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'church, church building', 'cinema, movie theater, movie theatre, movie house, picture palace', 'desktop computer', 'dishwasher, dish washer, dishwashing machine', 'forklift', 'fountain', 'iPod', 'minivan', 'mosque', 'parachute, chute', 'pickup, pickup truck', 'pillow', 'refrigerator, icebox', 'remote control, remote', 'restaurant, eating house, eating place, eatery', 'school bus', 'scoreboard', 'screen, CRT screen', 'shopping cart', 'stove', 'sunglasses, dark glasses, shades', 'syringe', 'table lamp', 'tank, army tank, armored combat vehicle, armoured combat vehicle', 'teapot', 'teddy, teddy bear', 'television, television system', 'toaster', 'toilet seat', 'torch', 'tractor', 'umbrella', 'vacuum, vacuum cleaner', 'vending machine', 'volleyball', 'wallet, billfold, notecase, pocketbook', 'wardrobe, closet, press', 'water bottle', 'water tower', 'web site, website, internet site, site', 'comic book', 'street sign', 'traffic light, traffic signal, stoplight', 'ice cream, icecream', 'cheeseburger', 'hotdog, hot dog, red hot', 'broccoli', 'cucumber, cuke', 'mushroom', 'strawberry', 'orange', 'lemon', 'pineapple, ananas', 'banana', 'pizza, pizza pie', 'burrito', 'red wine', 'espresso', 'seashore, coast, seacoast, sea-coast', 'volcano', 'ballplayer, baseball player', 'scuba diver']
skipped_sent = 0


classes_cifar = [
    'airplane',
    'car', #'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]


if DATASET == 'cifar10':
    plurals = ['s', 's', 's']
else:
    plurals = ['s', 'es', 'ies']



if OLD:
    path = "/l/users/20020067/Datasets/UMBC_corpus/imagenet_sentences_easy_sent1000_rand1678705786.tsv"
    # 100: "/l/users/20020067/Datasets/UMBC_corpus/imagenet_sentences_easy_sent100_rand1678543337.tsv"
    # 1000: "/l/users/20020067/Datasets/UMBC_corpus/imagenet_sentences_easy_sent1000_rand1678705786.tsv"
else:
    path = "/l/users/20020067/Datasets/UMBC_corpus/" + str(DATASET) + "_sentences_"  + str(DIFFICULTY) + "_sent" + str(NUM_OF_SENTENCES) + "_rand" + str(RAND) + ".tsv"

#with open("/l/users/20020067/Datasets/UMBC_corpus/imagenet_sentences_" + str(DIFFICULTY) + "_sent" + str(NUM_OF_SENTENCES) + "_rand" + str(RAND) + ".tsv", "r") as input:
with open(path, "r") as input:
    with open("/l/users/20020067/Datasets/UMBC_corpus/" + str(DATASET) + "_"  + str(DIFFICULTY) + "_" + "sent" + str(NUM_OF_SENTENCES) + "_" + str(DIVISION) + "_rand" + str(epoch_time) + ".tsv", "w") as output:

            lines = [line for line in input]
            
            classes = [line.split('\t')[0] for line in lines]

            paragraphs = [line.split('\t')[1] for line in lines]

            #print(classes[0])
            #print(paragraphs[0])

            for idx, paragraph in enumerate(paragraphs):
                #print(line)

                sentences = paragraph.split(' . ')

                sentence = []

                for sent_temp in sentences:

                    if ONE_CLASS:
                        classes_mid_100_list_all = []
                        
                        for cls in classes_mid_100_list:
                            if cls != classes[idx]:
                                if OPTIONS:
                                    cls_item_list = cls.split(", ")
                                else:
                                    cls_item_list = [cls]

                                for cls_item in cls_item_list:
                                    classes_mid_100_list_all.append(cls_item)
                        #print(classes_mid_100_list_all)

                    if OPTIONS:
                        classes_list = classes[idx].split(", ")
                    else:
                        classes_list = [classes[idx]]

                    for classes_item in classes_list:
                        #print(classes_item)
                        classes_item = classes_item.lower()

                        #if classes[idx] in sent_temp:
                        if classes_item in sent_temp:

                            if OLD:
                                #sent_temp.replace("b'", "")
                                sent_temp = sent_temp.replace("'","")
                                sent_temp = sent_temp[1:]
                            sent_temp = sent_temp.replace("\n"," ")

                            if DIVISION == 'Sent':
                                sentence.append(sent_temp) 
                                break

                            else:
                                words_num = int(DIVISION[0])

                                words = sent_temp.split(' ')
                                classes_item_list = classes_item.split(' ')

                                if STRICT:
                                    strict_multi = 0

                                for idw, word in enumerate(words):
                                    #if classes[idx] in word:
                                    if len(classes_item_list) > 1:
                                        if STRICT:
                                            classes_item_temp = classes_item_list[strict_multi]
                                            if len(classes_item_temp) <= STRICT_SYMB:
                                                word_match = False
                                                #if (classes_item_temp == word) or ((classes_item_temp +'s') == word) or ((classes_item_temp +'es') == word) or ((classes_item_temp[:-1] +'ies') == word):
                                                if (classes_item_temp == word) or ((classes_item_temp + plurals[0]) == word) or ((classes_item_temp + plurals[1]) == word) or ((classes_item_temp[:-1] + plurals[2]) == word):                                                
                                                    word_match = True

                                                    strict_multi += 1
                                                    if strict_multi == len(classes_item_list):
                                                        idw = idw - strict_multi + 1
                                                        strict_multi = 0
                                                        pass
                                                    else:
                                                        continue
                                                else:
                                                    word_match = False
                                                    strict_multi = 0
                                                    # break
                                                    continue
                                            else:
                                                strict_multi += 1
                                                if strict_multi == len(classes_item_list):
                                                    idw = idw - strict_multi + 1
                                                    strict_multi = 0
                                                    pass
                                                else:
                                                    continue

                                        else:
                                            classes_item_temp = classes_item_list[0]
                                            if classes_item_temp in word:
                                                pass
                                            else:
                                                continue

                                    else:
                                        classes_item_temp = classes_item

                                        if classes_item_temp in word:

                                            if STRICT:
                                                if len(classes_item_temp) <= STRICT_SYMB:
                                                    word_match = False
                                                    #if (classes_item_temp == word) or ((classes_item_temp +'s') == word) or ((classes_item_temp +'es') == word) or ((classes_item_temp[:-1] +'ies') == word):
                                                    if (classes_item_temp == word) or ((classes_item_temp + plurals[0]) == word) or ((classes_item_temp + plurals[1]) == word) or ((classes_item_temp[:-1] + plurals[2]) == word):
                                                        word_match = True

                                                    if word_match == True:
                                                        pass
                                                    else:
                                                        # break
                                                        continue
                                        else:
                                            continue

                                    # classes_temp = classes
                                    # classes_temp = classes.pop(idx)
                                    word_ind = idw
                                    #print(word_ind)
                                    
                                    start = - words_num + word_ind
                                    if start < 0:
                                        start = 0

                                    end = words_num + word_ind + len(classes_item_list) # cuz end not included
                                    if end > len(words):
                                        end = len(words)

                                    sentence = words[start : end]

                                    if ONE_CLASS:
                                        for foreign_cls in classes_mid_100_list_all:
                                            for word_i in sentence:
                                                #if (foreign_cls == word_i) or ((foreign_cls +'s') == word_i) or ((foreign_cls +'es') == word_i) or ((foreign_cls[:-1] +'ies') == word_i):
                                                if (foreign_cls == word_i) or ((foreign_cls + plurals[0]) == word_i) or ((foreign_cls + plurals[1]) == word_i) or ((foreign_cls[:-1] + plurals[2]) == word_i):                                                
                                                    skipped_sent +=1
                                                    #print(sentence)
                                                    #print(word_i)
                                                    word_id = sentence.index(word_i)
                                                    sentence.pop(word_id)
                                                    #print(sentence)
                                                    continue

                                    sentence = ' '.join(sentence) #.encode('utf-8').strip()
                                    sentence = str(sentence)

                                    if len(sentence) >= 5:
                                        output.write(classes[idx] + '\t')
                                        #output.write(classes_item + '\t')
                                        #output.write(classes_list[0] + '\t')

                                        output.write(sentence + '\n')
                                        # print(sentence)
                                        # print(len(sentence))
                                        # print(idx)
                                        #output.write('\n')
                                        #count += 1

                                break
                            #break

                        '''
                        elif DIVISION == '10words':
                            words = sent_temp.split(' ')
                            #print(words)

                            # word_ind = words.index(classes[idx])
                            # sentence = sentence[(-5 + word_ind) : word_ind : (5 + word_ind)]             

                            if STRICT:
                                strict_multi = 0
                            for idw, word in enumerate(words):
                                #if classes[idx] in word:
                                classes_item_list = classes_item.split(' ')
                                if len(classes_item_list) > 1:
                                    if STRICT:
                                        for classes_item_temp in classes_item_list:
                                            if len(classes_item_temp) <= 6:
                                                words = sentence.split(' ')        

                                                word_match = False
                                                for word in words:
                                                    if (classes_item_temp == word) or ((classes_item_temp +'s') == word) or ((classes_item_temp +'es') == word):
                                                        word_match = True
                                                        #print("_______________________yes")
                                                if word_match == True:
                                                    strict_multi += 1
                                                    if strict_multi == len(classes_item_list):
                                                        idw = idw - strict_multi
                                                        pass
                                                    else:
                                                        continue
                                                else:
                                                    # break
                                                    strict_multi = 0
                                                    continue
                                    else:
                                        classes_item_temp = classes_item_list[0]
                                        if classes_item_temp in word:
                                            pass
                                        else:
                                            continue

                                else:
                                    classes_item_temp = classes_item

                                    if classes_item_temp in word:

                                        if STRICT:
                                            if len(classes_item_temp) <= 6:
                                                words = sentence.split(' ')

                                                word_match = False
                                                for word in words:
                                                    if (classes_item_temp == word) or ((classes_item_temp +'s') == word) or ((classes_item_temp +'es') == word):
                                                        word_match = True
                                                        #print("_______________________yes")
                                                if word_match == True:
                                                    pass
                                                else:
                                                    # break
                                                    continue
                                    else:
                                        continue

                                # classes_temp = classes
                                # classes_temp = classes.pop(idx)
                                word_ind = idw
                                #print(word_ind)
                                
                                start = -10 + word_ind
                                if start < 0:
                                    start = 0

                                end = 10 + word_ind + len(classes_item_list) # cuz end not included
                                if end > len(words):
                                    end = len(words)

                                sentence = words[start : end]
                                sentence = ' '.join(sentence) #.encode('utf-8').strip()
                                sentence = str(sentence)
                            break


                        
                        elif DIVISION == '6words':
                            words = sent_temp.split(' ')
                            #print(words)

                            # word_ind = words.index(classes[idx])
                            # sentence = sentence[(-5 + word_ind) : word_ind : (5 + word_ind)]             

                            for idw, word in enumerate(words):
                                #if classes[idx] in word:
                                classes_item_list = classes_item.split(' ')
                                if len(classes_item_list) > 1:
                                    classes_item_temp = classes_item_list[0]
                                else:
                                    classes_item_temp = classes_item

                                if classes_item_temp in word:

                                    # classes_temp = classes
                                    # classes_temp = classes.pop(idx)
                                    word_ind = idw
                                    #print(word_ind)
                                    
                                    start = -6 + word_ind
                                    if start < 0:
                                        start = 0

                                    end = 6 + word_ind + len(classes_item_list) # cuz end not included
                                    if end > len(words):
                                        end = len(words)

                                    sentence = words[start : end]
                                    sentence = ' '.join(sentence) #.encode('utf-8').strip()
                                    sentence = str(sentence)
                            break



                        elif DIVISION == '5words':
                            words = sent_temp.split(' ')
                            #print(words)

                            # word_ind = words.index(classes[idx])
                            # sentence = sentence[(-5 + word_ind) : word_ind : (5 + word_ind)]             

                            if STRICT:
                                strict_multi = 0
                            for idw, word in enumerate(words):
                                #if classes[idx] in word:
                                classes_item_list = classes_item.split(' ')
                                if len(classes_item_list) > 1:
                                    if STRICT:
                                        for classes_item_temp in classes_item_list:
                                            if len(classes_item_temp) <= 6:
                                                word_match = False
                                                if (classes_item_temp == word) or ((classes_item_temp +'s') == word) or ((classes_item_temp +'es') == word):
                                                    word_match = True

                                                if word_match == True:
                                                    strict_multi += 1
                                                    if strict_multi == len(classes_item_list):
                                                        idw = idw - strict_multi
                                                        pass
                                                    else:
                                                        continue
                                                else:
                                                    # break
                                                    strict_multi = 0
                                                    continue
                                    else:
                                        classes_item_temp = classes_item_list[0]
                                        if classes_item_temp in word:
                                            pass
                                        else:
                                            continue

                                else:
                                    classes_item_temp = classes_item

                                    if classes_item_temp in word:

                                        if STRICT:
                                            if len(classes_item_temp) <= 6:
                                                word_match = False
                                                if (classes_item_temp == word) or ((classes_item_temp +'s') == word) or ((classes_item_temp +'es') == word):
                                                    word_match = True

                                                if word_match == True:
                                                    pass
                                                else:
                                                    # break
                                                    continue
                                    else:
                                        continue

                                # classes_temp = classes
                                # classes_temp = classes.pop(idx)
                                word_ind = idw
                                #print(word_ind)
                                
                                start = -5 + word_ind
                                if start < 0:
                                    start = 0

                                end = 5 + word_ind + len(classes_item_list) # cuz end not included
                                if end > len(words):
                                    end = len(words)

                                sentence = words[start : end]
                                sentence = ' '.join(sentence) #.encode('utf-8').strip()
                                sentence = str(sentence)
                            break

                        
                        # elif DIVISION == '5words':
                        #     words = sent_temp.split(' ')
                        #     #print(words)

                        #     # word_ind = words.index(classes[idx])
                        #     # sentence = sentence[(-5 + word_ind) : word_ind : (5 + word_ind)]             

                        #     for idw, word in enumerate(words):
                        #         #if classes[idx] in word:
                        #         classes_item_list = classes_item.split(' ')
                        #         if len(classes_item_list) > 1:
                        #             classes_item_temp = classes_item_list[0]
                        #         else:
                        #             classes_item_temp = classes_item

                        #         if classes_item_temp in word:

                        #             # classes_temp = classes
                        #             # classes_temp = classes.pop(idx)
                        #             word_ind = idw
                        #             #print(word_ind)
                                    
                        #             start = -5 + word_ind
                        #             if start < 0:
                        #                 start = 0

                        #             end = 5 + word_ind + len(classes_item_list) # cuz end not included
                        #             if end > len(words):
                        #                 end = len(words)

                        #             sentence = words[start : end]
                        #             sentence = ' '.join(sentence) #.encode('utf-8').strip()
                        #             sentence = str(sentence)
                        #     break
                        

                        elif DIVISION == '4words':
                            words = sent_temp.split(' ')
                            #print(words)

                            # word_ind = words.index(classes[idx])
                            # sentence = sentence[(-5 + word_ind) : word_ind : (5 + word_ind)]             

                            for idw, word in enumerate(words):
                                #if classes[idx] in word:
                                classes_item_list = classes_item.split(' ')
                                if len(classes_item_list) > 1:
                                    classes_item_temp = classes_item_list[0]
                                else:
                                    classes_item_temp = classes_item

                                if classes_item_temp in word:

                                    # classes_temp = classes
                                    # classes_temp = classes.pop(idx)
                                    word_ind = idw
                                    #print(word_ind)
                                    
                                    start = -4 + word_ind
                                    if start < 0:
                                        start = 0

                                    end = 4 + word_ind + len(classes_item_list) # cuz end not included
                                    if end > len(words):
                                        end = len(words)

                                    sentence = words[start : end]
                                    sentence = ' '.join(sentence) #.encode('utf-8').strip()
                                    sentence = str(sentence)
                            break


                        elif DIVISION == '3words':
                            words = sent_temp.split(' ')
                            #print(words)

                            # word_ind = words.index(classes[idx])
                            # sentence = sentence[(-5 + word_ind) : word_ind : (5 + word_ind)]             

                            for idw, word in enumerate(words):
                                #if classes[idx] in word:
                                classes_item_list = classes_item.split(' ')
                                if len(classes_item_list) > 1:
                                    classes_item_temp = classes_item_list[0]
                                else:
                                    classes_item_temp = classes_item

                                if classes_item_temp in word:

                                    # classes_temp = classes
                                    # classes_temp = classes.pop(idx)
                                    word_ind = idw
                                    #print(word_ind)
                                    
                                    start = -3 + word_ind
                                    if start < 0:
                                        start = 0

                                    end = 3 + word_ind + len(classes_item_list) # cuz end not included
                                    if end > len(words):
                                        end = len(words)

                                    sentence = words[start : end]
                                    sentence = ' '.join(sentence) #.encode('utf-8').strip()
                                    sentence = str(sentence)
                            break


                        elif DIVISION == '2words':
                            words = sent_temp.split(' ')
                            #print(words)

                            # word_ind = words.index(classes[idx])
                            # sentence = sentence[(-5 + word_ind) : word_ind : (5 + word_ind)]             

                            for idw, word in enumerate(words):
                                #if classes[idx] in word:
                                classes_item_list = classes_item.split(' ')
                                if len(classes_item_list) > 1:
                                    classes_item_temp = classes_item_list[0]
                                else:
                                    classes_item_temp = classes_item

                                if classes_item_temp in word:

                                    # classes_temp = classes
                                    # classes_temp = classes.pop(idx)
                                    word_ind = idw
                                    #print(word_ind)
                                    
                                    start = -2 + word_ind
                                    if start < 0:
                                        start = 0

                                    end = 2 + word_ind + len(classes_item_list) # cuz end not included
                                    if end > len(words):
                                        end = len(words)

                                    sentence = words[start : end]
                                    sentence = ' '.join(sentence) #.encode('utf-8').strip()
                                    sentence = str(sentence)
                            break


                        elif DIVISION == '1words':

                            words = sent_temp.split(' ')
                            #print(words)

                            # word_ind = words.index(classes[idx])
                            # sentence = sentence[(-5 + word_ind) : word_ind : (5 + word_ind)]             

                            for idw, word in enumerate(words):
                                #if classes[idx] in word:
                                classes_item_list = classes_item.split(' ')
                                if len(classes_item_list) > 1:
                                    classes_item_temp = classes_item_list[0]
                                else:
                                    classes_item_temp = classes_item

                                if classes_item_temp in word:

                                    # classes_temp = classes
                                    # classes_temp = classes.pop(idx)
                                    word_ind = idw
                                    #print(word_ind)
                                    
                                    start = -1 + word_ind
                                    if start < 0:
                                        start = 0

                                    end = 1 + word_ind + len(classes_item_list) # cuz end not included
                                    if end > len(words):
                                        end = len(words)

                                    sentence = words[start : end]
                                    sentence = ' '.join(sentence) #.encode('utf-8').strip()
                                    sentence = str(sentence)
                            break
                        '''
                        #break

                if DIVISION == 'Sent':
                    #sentence = ' '.join(sentence)
                    #if len(sentence) >= 5:
                    sentence = ' . '.join(sentence) #.encode('utf-8').strip()
                    sentence = str(sentence)
                    sentence = sentence.replace("  "," ")


                    if len(sentence) >= 5:
                        output.write(classes[idx] + '\t')
                        #output.write(classes_item + '\t')
                        #output.write(classes_list[0] + '\t')

                        output.write(sentence + '\n')
                        # print(sentence)
                        # print(len(sentence))
                        # print(idx)
                        #output.write('\n')
                        #count += 1
            
print("Foreign class occurencies: ", skipped_sent)
