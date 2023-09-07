import numpy as np
from utils import get_similarities_per_class
import torch
import clip 
# or below line
from Project.CLIP.CLIP import clip
from tqdm.notebook import tqdm
from pkg_resources import packaging
import pathlib
import tarfile
import requests
import shutil
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
# ! pip install git+https://github.com/modestyachts/ImageNetV2_pytorch
from ImagenetData_Class import ImageNetV2Dataset_Our, ImageNetValDataset
import torchvision
import torch
import torchvision.transforms as transforms
from utils import *
import torch
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from captum.attr import visualization
print("Torch version:", torch.__version__)

# clip.available_models()
model, preprocess = clip.load("ViT-B/32")

model.visual.input_resolution = 32 # 224 originally
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size
print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]
print(f"{len(imagenet_classes)} classes, {len(imagenet_templates)} templates")

# from imagenetv2_pytorch import ImageNetV2Dataset
images = ImageNetV2Dataset_Our(transform=preprocess)
try:
  images = torchvision.datasets.ImageNet("/content/", split='val', transform=preprocess)
except:
  print("Ignore this error")
loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=2)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
classes = ('airplane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

imagenet_classes = [
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
imagenet_templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

auxillary = []
zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates, model)

'''
## 12 classes (easy):

imagenet_classes_12 = ["cucumber", "mushroom", "banana", "pizza", "bucket", "umbrella", "mailbox", "microwave oven", "rifle", "torch", "volleyball", "taxicab"]
imagenet_indices_12 = []

for class_name in imagenet_classes_12:
  ind = imagenet_classes.index(class_name)
  imagenet_indices_12.append(ind)
print(imagenet_indices_12)

print(f"{len(imagenet_classes_12)} classes, {len(imagenet_templates)} templates")
'''


## 100 classes (mid):
imagenet_classes_100 = []

# Old (for files with non-strict, strict, with class repeat)
# imagenet_classes_100_our = ['goldfish, Carassius auratus', 'hammerhead, hammerhead shark', 'cock', 'hen', 'bald eagle, American eagle, Haliaeetus leucocephalus', 'scorpion', 'garden spider, Aranea diademata', 'black widow, Latrodectus mactans', 'tarantula', 'wolf spider, hunting spider', 'hummingbird', 'goose', 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 'wombat', 'jellyfish', 'flamingo', 'pelican', 'albatross, mollymawk', 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus', 'sea lion', 'Chihuahua', 'toy terrier', 'Scotch terrier, Scottish terrier, Scottie', 'German shepherd, German shepherd dog, German police dog, alsatian', 'Doberman, Doberman pinscher', 'Siberian husky', 'coyote, prairie wolf, brush wolf, Canis latrans', 'red fox, Vulpes vulpes', 'lion, king of beasts, Panthera leo', 'tiger, Panthera tigris', 'cheetah, chetah, Acinonyx jubatus', 'zebra', 'bison', 'gorilla, Gorilla gorilla', 'chimpanzee, chimp, Pan troglodytes', 'assault rifle, assault gun', 'backpack, back pack, knapsack, packsack, rucksack, haversack', 'cannon', 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM', 'castle', 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'church, church building', 'cinema, movie theater, movie theatre, movie house, picture palace', 'desktop computer', 'dishwasher, dish washer, dishwashing machine', 'forklift', 'fountain', 'iPod', 'parachute, chute', 'pickup, pickup truck', 'pillow', 'refrigerator, icebox', 'remote control, remote', 'restaurant, eating house, eating place, eatery', 'school bus', 'scoreboard', 'screen, CRT screen', 'shopping cart', 'stove', 'sunglasses, dark glasses, shades', 'syringe', 'table lamp', 'tank, army tank, armored combat vehicle, armoured combat vehicle', 'teapot', 'teddy, teddy bear', 'television, television system', 'toaster', 'toilet seat', 'torch', 'tractor', 'umbrella', 'vacuum, vacuum cleaner', 'vending machine', 'volleyball', 'wallet, billfold, notecase, pocketbook', 'wardrobe, closet, press', 'water bottle', 'water tower', 'web site, website, internet site, site', 'comic book', 'street sign', 'traffic light, traffic signal, stoplight', 'ice cream, icecream', 'cheeseburger', 'hotdog, hot dog, red hot', 'broccoli', 'cucumber, cuke', 'mushroom', 'strawberry', 'orange', 'lemon', 'pineapple, ananas', 'banana', 'pizza, pizza pie', 'burrito', 'red wine', 'espresso', 'seashore, coast, seacoast, sea-coast', 'volcano', 'ballplayer, baseball player']
# imagenet_indices_100 = [1, 4, 7, 8, 22, 71, 74, 75, 76, 77, 94, 99, 105, 106, 107, 130, 144, 146, 147, 150, 151, 158, 199, 235, 236, 250, 272, 277, 291, 292, 293, 340, 347, 366, 367, 413, 414, 471, 480, 483, 487, 497, 498, 527, 534, 561, 562, 605, 701, 717, 721, 760, 761, 762, 779, 781, 782, 791, 827, 837, 845, 846, 847, 849, 850, 851, 859, 861, 862, 866, 879, 882, 886, 890, 893, 894, 898, 900, 916, 917, 919, 920, 928, 933, 934, 937, 943, 947, 949, 950, 951, 953, 954, 963, 965, 966, 967, 978, 980, 981]

# New (replace toy terrier, garden spider, Siberian Husky)
imagenet_classes_100_our = ['goldfish, Carassius auratus', 'hammerhead, hammerhead shark', 'cock', 'hen', 'bald eagle, American eagle, Haliaeetus leucocephalus', 'scorpion', 'black widow, Latrodectus mactans', 'tarantula', 'wolf spider, hunting spider', 'hummingbird', 'goose', 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 'wombat', 'jellyfish', 'flamingo', 'pelican', 'albatross, mollymawk', 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus', 'sea lion', 'Chihuahua', 'Scotch terrier, Scottish terrier, Scottie', 'German shepherd, German shepherd dog, German police dog, alsatian', 'Doberman, Doberman pinscher', 'coyote, prairie wolf, brush wolf, Canis latrans', 'red fox, Vulpes vulpes', 'lion, king of beasts, Panthera leo', 'tiger, Panthera tigris', 'cheetah, chetah, Acinonyx jubatus', 'zebra', 'bison', 'gorilla, Gorilla gorilla', 'chimpanzee, chimp, Pan troglodytes', 'assault rifle, assault gun', 'backpack, back pack, knapsack, packsack, rucksack, haversack', 'cannon', 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM', 'castle', 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'church, church building', 'cinema, movie theater, movie theatre, movie house, picture palace', 'desktop computer', 'dishwasher, dish washer, dishwashing machine', 'forklift', 'fountain', 'iPod', 'minivan', 'mosque', 'parachute, chute', 'pickup, pickup truck', 'pillow', 'refrigerator, icebox', 'remote control, remote', 'restaurant, eating house, eating place, eatery', 'school bus', 'scoreboard', 'screen, CRT screen', 'shopping cart', 'stove', 'sunglasses, dark glasses, shades', 'syringe', 'table lamp', 'tank, army tank, armored combat vehicle, armoured combat vehicle', 'teapot', 'teddy, teddy bear', 'television, television system', 'toaster', 'toilet seat', 'torch', 'tractor', 'umbrella', 'vacuum, vacuum cleaner', 'vending machine', 'volleyball', 'wallet, billfold, notecase, pocketbook', 'wardrobe, closet, press', 'water bottle', 'water tower', 'web site, website, internet site, site', 'comic book', 'street sign', 'traffic light, traffic signal, stoplight', 'ice cream, icecream', 'cheeseburger', 'hotdog, hot dog, red hot', 'broccoli', 'cucumber, cuke', 'mushroom', 'strawberry', 'orange', 'lemon', 'pineapple, ananas', 'banana', 'pizza, pizza pie', 'burrito', 'red wine', 'espresso', 'seashore, coast, seacoast, sea-coast', 'volcano', 'ballplayer, baseball player', 'scuba diver']
imagenet_indices_100 = [1, 4, 7, 8, 22, 71, 75, 76, 77, 94, 99, 105, 106, 107, 130, 144, 146, 147, 150, 151, 199, 235, 236, 272, 277, 291, 292, 293, 340, 347, 366, 367, 413, 414, 471, 480, 483, 487, 497, 498, 527, 534, 561, 562, 605, 656, 668, 701, 717, 721, 760, 761, 762, 779, 781, 782, 791, 827, 837, 845, 846, 847, 849, 850, 851, 859, 861, 862, 866, 879, 882, 886, 890, 893, 894, 898, 900, 916, 917, 919, 920, 928, 933, 934, 937, 943, 947, 949, 950, 951, 953, 954, 963, 965, 966, 967, 978, 980, 981, 983]
for class_index in imagenet_indices_100:
  cls = imagenet_classes[class_index]
  print(class_index)
  imagenet_classes_100.append(cls)
print(imagenet_classes_100)
print(f"{len(imagenet_classes_100)} classes, {len(imagenet_templates)} templates")

zeroshot_weights_100 = zeroshot_classifier(imagenet_classes_100, imagenet_templates, model)

from utils import top_acc
top1, top5 = top_acc(loader, model, zeroshot_weights)
print(f"Top-1 accuracy: {top1:.2f}")
print(f"Top-5 accuracy: {top5:.2f}")

top1, top5 = top_acc(testloader, model, zeroshot_weights)
print(f"Top-1 accuracy: {top1:.2f}")
print(f"Top-5 accuracy: {top5:.2f}")

start_layer =  -1 #@param {type:"number"}
start_layer_text =  -1 #@param {type:"number"}

# sample with ViT-B/32
#We can replace the model part
clip.clip._MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
img_path = "/home/raza.imam/Documents/ML708B/Transformer-MM-Explainability/CLIP/glasses.png"
img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
texts = ["a man with eyeglasses"]
text = clip.tokenize(texts).to(device)
#for loop:
print(img.shape, text.shape)
logits_per_image, logits_per_text = model(img, text)
print(color.BOLD + color.PURPLE + color.UNDERLINE + f'CLIP similarity score: {logits_per_image.item()}' + color.END)
R_text, R_image = interpret(model=model, image=img, texts=text, device=device, start_layer=start_layer, start_layer_text=start_layer_text)
batch_size = text.shape[0]
for i in range(batch_size):
  show_heatmap_on_text(texts[i], text[i], R_text[i])
  show_image_relevance(R_image[i], img, orig_image=Image.open(img_path))
  plt.show()

#-------------Ours--------------

all_images_dict =image_loader(folder_path='clip_autoPrompt/Data/easy classes/Images/')
all_images = all_images_dict

# Example usage:
DIFFICULTY = 'mid'
PROMPTS_NUMBER = 100
STRICT = True
df_prompts10words, df_prompts5words, df_prompts3words, df_prompts1words, df_promptsSent = load_dataframes(DIFFICULTY, PROMPTS_NUMBER, STRICT)

if DIFFICULTY == 'easy':
  dictionary_df_prompts10words_repeat = get_class_prompt_dict(df_prompts10words)
  dictionary_df_prompts5words_repeat = get_class_prompt_dict(df_prompts5words)
  dictionary_df_prompts3words_repeat = get_class_prompt_dict(df_prompts3words)
  dictionary_df_prompts1words_repeat = get_class_prompt_dict(df_prompts1words)
if DIFFICULTY == 'mid':
  dictionary_df_prompts1words_repeat = get_class_prompt_dict(df_prompts1words)
  
new_dict = dictionary_df_prompts1words_repeat
for key in new_dict:
  if len(new_dict[key]) < PROMPTS_NUMBER:
    print(key, len(new_dict[key]))
    
# 1. Generating Dictionary of scores with first image of all classes and _prompts10words, and ranking them in next cell
dictionary_df_prompts10words = get_similarities_per_class(all_images, df_prompts10words)
for i in dictionary_df_prompts10words.keys():
  print("For class ", i)
  print(sort_dict_by_key(dictionary_df_prompts10words[i]))
  
for key in dictionary_df_prompts10words:
  if len(dictionary_df_prompts10words[key]) < PROMPTS_NUMBER:
    print(key, len(new_dict[key]))
    
# 2. Generating Dictionary of scores with first image of all classes and _prompts5words, and ranking them in next cell
all_images['cucumber'][0].shape # We are only trying for first image of the cucumber class
print(df_prompts5words.groupby(0).head(1))
dictionary_df_prompts5words = get_similarities_per_class(all_images, df_prompts5words)
print(dictionary_df_prompts5words) #df_prompts5words

for i in dictionary_df_prompts5words.keys():
  print("For class ", i)
  print(sort_dict_by_key(dictionary_df_prompts5words[i]))
  
for key in dictionary_df_prompts5words:
  if len(dictionary_df_prompts5words[key]) < PROMPTS_NUMBER:
    print(key, len(new_dict[key]))
    

imagenet_classes_100temp = imagenet_classes_100
auxillary = get_prompts(imagenet_classes_100temp, imagenet_templates)
print(auxillary)

print(auxillary.keys()) #80 prompts for cucumber class
imagenet_df = pd.DataFrame(list(auxillary.items()), columns=[0, 1])
imagenet_df = imagenet_df.explode(1)
imagenet_df

all_images['cucumber'][0].shape # We are only trying for first image of the first class (cucumber class)
dictionary_imagenet_df = get_similarities_per_class(all_images, imagenet_df)
print(dictionary_imagenet_df) #df_prompts5words

for i in dictionary_imagenet_df.keys():
  print("For class ", i)
  print(sort_dict_by_key(dictionary_imagenet_df[i]))
  
for key in dictionary_imagenet_df:
  if len(dictionary_imagenet_df[key]) < PROMPTS_NUMBER:
    print(key, len(new_dict[key]))
    
dictionary_imagenet_df_repeat = get_class_prompt_dict(imagenet_df)

import statistics
def results(dictionary_df):
  df_keys = list(dictionary_df.keys())
  for df_class in df_keys:
    print("\nFor Class ", df_class, "-")
    d = dictionary_df[df_class]
    keys = list(d.keys())
    values = list(d.values())
    position_min = values.index(min(values))
    position_max = values.index(max(values))
    print("Minimum score:", min(values), "with Prompt:", keys[position_min])
    print("Maximum score:", max(values), "with Prompt:", keys[position_max])
    print("Mean score:", statistics.mean(values))
    print("Median score:", statistics.median(values))
print("\nFor _prompt2words------------------------------>>>:")
results(dictionary_df_prompts10words)