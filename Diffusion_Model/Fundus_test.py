from diffusers import StableDiffusionPipeline
from torchvision import transforms

import torch, os
from PIL import Image
import numpy as np
import pandas as pd

from LANet import resnet_lanet, densenet_lanet, vgg_lanet, inceptionv3_lanet

class_dict = {
    #'normal': (0, "./sd-retina-model-lora-fundus_normal/checkpoint-1000"), 
    #'mild DR': (1, "./1_0p1_0p001/sd-retina-model-lora-fundus/checkpoint-1000"), 
    'moderate DR': (2, "./sd-retina-model-lora-fundus2_mod2/checkpoint-1000"), 
    #'severe DR': (3, "./1_0p1_0p001/sd-retina-model-lora-fundus2/checkpoint-1500"),
    #'proliferative DR': (4, "./sd-retina-model-lora-fundus_PDR/checkpoint-1000")
    }
outdir = '/home/haochen/DataComplex/universe/DataSet/Retina_fundus_test'
num_run = 7000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_load_pretrained_classifiers(model, adaloss=True):
    print(model)
    # Initialize network based on the selected model
    if  'res50' in model:
        net = resnet_lanet.resnet50(pretrained=False, adaloss=adaloss)
    elif 'dense121' in model:
        net = densenet_lanet.densenet121(pretrained=False, adaloss=adaloss)
        net.classifier = torch.nn.Linear(1024, 5)
    elif 'vgg' in model:
        net = vgg_lanet.vgg16_bn(pretrained=False, adaloss=adaloss)
        net.classifier[6] = torch.nn.Linear(4096, 5)
    elif 'inceptionv3' in model:
        #net = inceptionv3_lanet.inception_v3(pretrained=True, aux_logits=False, adaloss=adaloss)
        net = inceptionv3_lanet.inception_v3(pretrained=False, aux_logits=False, transform_input=True, adaloss=adaloss)
        net.fc = torch.nn.Linear(2048, 5)

    # Load the correct weights for the model
    ckpt_dir = '/home/haochen/DataComplex/universe/DataSet/diabet_fundus/LANet_checkpoints/'
    if model == 'res50':
        weight_dir = ckpt_dir + 'resnet50_LAM_AJL_197.pkl'
    elif model == 'vgg':
        weight_dir = ckpt_dir + 'vgg16_bn_LAM_AJL_65.pkl'
    elif model == 'dense121':
        weight_dir = ckpt_dir + 'densenet121_LAM_AJL_199.pkl'
    elif model == 'inceptionv3':
        weight_dir = ckpt_dir + 'inceptionv3_LAM_AJL_noaux_267.pkl'
    elif model == 'my_vgg_1':
        weight_dir = ckpt_dir + 'ddr_vgg_lanet_adl_f0_os/15.pkl'
    elif model == 'my_vgg_2':
        weight_dir = ckpt_dir + 'ddr_vgg_lanet_adl_f2_os/13.pkl'
    elif model == 'my_vgg_3':
        weight_dir = ckpt_dir + 'ddr_vgg_lanet_adl_f3_os/23.pkl'
    elif model == 'my_dense121':
        weight_dir = ckpt_dir + 'ddr_dense121_lanet_adl_f1_os/7.pkl'
    elif model == 'my_inceptionv3_1':
        weight_dir = ckpt_dir + 'ddr_inceptionv3_lanet_adl_seed2_f0_os/11.pkl'
    elif model == 'my_inceptionv3_2':
        weight_dir = ckpt_dir + 'ddr_inceptionv3_lanet_adl_seed3_f3_os/21.pkl'

    weights = torch.load(weight_dir, map_location=torch.device('cpu'))
    new_state_dict = {}
    for key, value in weights['net'].items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    net.load_state_dict(new_state_dict)
    net.requires_grad_(False)
    net = net.to('cpu')
    net.eval()
    return net
#classifiers = ['res50', 'dense121', 'vgg', 'inceptionv3']
classifiers = ['my_vgg_1', 'my_vgg_2', 'my_vgg_3', 'my_dense121', 'my_inceptionv3_1', 'my_inceptionv3_2']
classifier_model_list = [init_load_pretrained_classifiers(c) for c in classifiers]
print(len(classifier_model_list))

classifier_transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.426, .298, .213],std=[.277, .203, .169])
    ])

idx_list = []
score_list = []
label_list = []
csv_data = {
    'idx': [],
    'class': [],
    'class_id': [],
    'img_path': [],
    'npz_path': [],
    'score': [],
    'label': [],
}

idx = 10**6
for clss in class_dict:
    os.makedirs(os.path.join(outdir,clss), exist_ok=True)
    prompt = 'A color fundus image from DDR dataset in {} stage.'.format(clss)
    print(prompt)

    clss_idx = class_dict[clss][0]
    model_path = class_dict[clss][1]
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    #pipe = StableDiffusionPipeline.from_pretrained("Nihirc/Prompt2MedImage", torch_dtype=torch.float16)
    pipe.load_lora_weights(model_path)
    pipe.to(device)

    for _ in range(num_run):
        images = pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.5, num_images_per_prompt=16).images
        #images_batch = torch.stack([torch.tensor(classifier_transform(np.array(img).transpose(2, 0, 1))) for img in images]) # to numpy + transform
        images_batch = torch.stack([classifier_transform(img) for img in images])
        #print(images_batch.shape) #torch.Size([16, 3, 256, 256])
        images_batch = images_batch.to(device)
        likelihood_scores = torch.zeros(images_batch.shape[0])
        softlabel = torch.zeros(images_batch.shape[0], 5)
        for classifier in classifier_model_list:
            #print('Loading model from: {}'.format(mpath))
            classifier.to(device)
            _, classifier_outputs = classifier(images_batch.to(device))
            classifier_outputs = torch.softmax(classifier_outputs, dim=1).cpu()
            softlabel += classifier_outputs
            likelihood_scores += classifier_outputs[:, clss_idx]
            classifier = classifier.to('cpu')
        #likelihood_scores = likelihood_scores.numpy()
        softlabel = softlabel.numpy()
        #print(likelihood_scores)
        #print(softlabel[0])
        #print(softlabel[1])
        # Select top 8 samples with the best likelihood scores
        _, top_indices = likelihood_scores.topk(12, largest=True)
        #print(likelihood_scores)
        for top_indice in top_indices:
            if likelihood_scores[top_indice] < len(classifier_model_list)*0.5 or likelihood_scores[top_indice] > len(classifier_model_list)*0.95:
                continue
            idx+=1
            gray_image = images[top_indice]
            score = likelihood_scores[top_indice].item()
            label = softlabel[top_indice]
            label = label/np.sum(label)

            # save
            img_save_path = os.path.join(outdir,clss,"{:0>7d}.png".format(idx))
            gray_image.save(img_save_path)
            npz_save_path = os.path.join(outdir,clss,"{:0>7d}.npz".format(idx))
            np.savez(npz_save_path, idx=idx, score=score, label=label, annotation=clss_idx)
            # record
            csv_data['idx'].append("{:0>7d}".format(idx))
            csv_data['class'].append(clss)
            csv_data['class_id'].append(clss_idx)
            csv_data['img_path'].append(img_save_path)
            csv_data['npz_path'].append(npz_save_path)
            csv_data['score'].append(score)
            csv_data['label'].append("{}".format(label.tolist()))

df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(outdir, 'data.csv'), index=False)