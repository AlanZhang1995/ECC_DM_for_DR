import torch, os
import PIL

from torchvision import datasets, transforms

class SyncAMDDataset(torch.utils.data.Dataset):
    def __init__(self, root, tokenizer, transform=None):
        self.image_folder = datasets.ImageFolder(root, transform=transform)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        image, label = self.image_folder[idx]

        # Convert label to text prompt and tokenize it
        label_text = self.image_folder.classes[label]
        label_prompt = "A color fundus image from DDR dataset in {} stage.".format(label_text)
        #print(label_prompt)
        inputs = self.tokenizer(
            label_prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return dict(pixel_values=image, input_ids=inputs.input_ids)


# ---------- 3fundus Dataset + ddr Dataset  ----------
def labeltag2prompt(label, tag, instance=False, th=0.7):
    label2stage = {
        0: 'normal',
        1: 'mild DR',
        2: 'moderate DR',
        3: 'severe DR',
        4: 'proliferative DR'
    }
    stage = label2stage[label] if label<5 else None
    if instance:
        prompt = 'A color fundus image from {} dataset in {} stage.'.format(tag, stage)
        return prompt
    prompt = 'A color fundus image'
    if torch.rand(1)<th:
        prompt += ' from {} dataset'.format(tag)
    if stage is not None and torch.rand(1)<th:
        prompt += ' in {} stage'.format(stage)
    return prompt+'.'

    
class CSVDataset_Fundus_3combine(torch.utils.data.Dataset):
    def __init__(self, csv_file, tokenizer, stage=None, transform=None):
        self.images = []
        self.ddr_labeled_image = []
        #self.labels = []
        #self.scale = 700
        self.csv_file = csv_file
        self.tokenizer = tokenizer
        #self.root_dir = root_dir #useless
        self.eyepacs_path = '/home/haochen/DataComplex/universe/DataSet/diabet_fundus/EyePacs'
        self.aptos_path = '/home/haochen/DataComplex/universe/DataSet/diabet_fundus/aptos2019-blindness-detection'
        self.ddr_path = '/home/haochen/DataComplex/universe/DataSet/diabet_fundus/DDR-dataset/DR_grading'
        self.root_dirs = {
            'EyePacs': self.eyepacs_path + '/train_preprocessed1024/{}.jpeg',
            'APTOS': self.aptos_path + '/train_images_preprocessed1024/{}.png',
            'DDR': self.ddr_path + '/all_data_preprocessed1024/{}',
        }
        self.stage = stage
        self.transform = transform
        for index, row in csv_file.iterrows():
            # prior term
            tag = row['dataset']
            #img_name = os.path.join(self.root_dir, row['image']+'.jpeg')
            img_name = self.root_dirs[tag].format(row['image'])
            label = int(row['level'])
            self.images.append([img_name, label, tag])

            # MSE term
            if tag == 'DDR' and (stage is None or label == stage):
                self.ddr_labeled_image.append([img_name, label, tag])
        print('3dataset_labeled #:', len(self.images))

        # EyePacs unlabeled
        imgs_eyepacs_unlabeled = self.get_image_paths(folder_path=os.path.join(self.eyepacs_path, 'test_preprocessed1024'), tag='EyePacs')
        print('eyepacs_unlabeled #:', len(imgs_eyepacs_unlabeled))

        # APTOS unlabeled
        imgs_aptos_unlabeled = self.get_image_paths(folder_path=os.path.join(self.aptos_path, 'test_images_preprocessed1024'), tag='APTOS')
        print('aptos_unlabeled #:', len(imgs_aptos_unlabeled))

        # DDR unlabeled
        imgs_ddr_unlabeled = []
        with open(os.path.join(self.ddr_path, 'unlabeled.txt'), encoding='utf-8') as file:
            for line in file.readlines():
                line = line.strip('\n')
                img, label = line.split('\t')
                label = int(label)
                assert label == 5
                imgs_ddr_unlabeled.append([os.path.join(self.ddr_path, 'all_data_preprocessed1024', img), label, 'DDR'])
        print('ddr_unlabeled #:', len(imgs_ddr_unlabeled))

        self.images = self.images + imgs_eyepacs_unlabeled + imgs_aptos_unlabeled + imgs_ddr_unlabeled
        print('Total #:', len(self.images))
        self.ddr_len = len(self.ddr_labeled_image)
        print('Total DDR (stage={}) #: {}'.format(stage, self.ddr_len))

    def __len__(self):
        return max(self.ddr_len, len(self.images))

    def get_image_paths(self, folder_path, tag):
        # List of supported image file extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
        image_paths = []
        # Iterate over files in the folder
        for root, _, files in os.walk(folder_path):
            for file in files:
                # Check file extension
                if os.path.splitext(file)[1].lower() in image_extensions:
                    # Append full path to the list
                    image_paths.append([os.path.join(root, file), 5, tag])
        return image_paths

    def __getitem__(self, index):
        img, label, tag = self.images[index]
        class_image = PIL.Image.open(img).convert('RGB')
        #try:
        #    class_image = PIL.Image.open(img).convert('RGB')
        #except:
        #    # Handle the case where the image file is truncated or cannot be opened
        #    print(f"Skipping truncated image: {item}")
        #    class_image = np.zeros((self.hr_shape, self.hr_shape, 3))
        if self.transform:
            class_image = self.transform(class_image)
        class_prompt = labeltag2prompt(label, tag, instance=False)
        class_text_inputs = self.tokenizer(
            class_prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        img, label, tag = self.ddr_labeled_image[index % self.ddr_len]
        instance_images = PIL.Image.open(img).convert('RGB')
        if self.transform:
            instance_images = self.transform(instance_images)
        instance_prompt = labeltag2prompt(label, tag, instance=True)
        instance_text_inputs = self.tokenizer(
            instance_prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        #print('class_prompt: ', class_prompt)
        #print('instance_prompt: ', instance_prompt, label, tag)

        return dict(instance_images=instance_images, instance_prompt_ids=instance_text_inputs.input_ids, class_images=class_image, class_prompt_ids=class_text_inputs.input_ids)

class DDR_test_set(torch.utils.data.Dataset):
    def __init__(self, csv_file, tokenizer, stage=None, transform=None):
        self.images = []
        self.csv_file = csv_file
        self.transform = transform
        self.ddr_path = '/home/haochen/DataComplex/universe/DataSet/diabet_fundus/DDR-dataset/DR_grading'

        # DDR unlabeled
        imgs_ddr_unlabeled = []
        with open(os.path.join(self.ddr_path, 'test.txt'), encoding='utf-8') as file:
            for line in file.readlines():
                line = line.strip('\n')
                img, label = line.split(' ') #\t
                label = int(label)
                if label!=5:
                    imgs_ddr_unlabeled.append([os.path.join(self.ddr_path, 'all_data_preprocessed1024', img), label, 'DDR'])
        print('ddr_unlabeled #:', len(imgs_ddr_unlabeled))

        self.images = self.images + imgs_ddr_unlabeled
        print('Total #:', len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, label, tag = self.images[index]
        class_image = PIL.Image.open(img).convert('RGB')
        if self.transform:
            class_image = self.transform(class_image)
        class_prompt = labeltag2prompt(label, tag, instance=False)
        #print('class_prompt: ', class_prompt)
        label_clf = 0 if label == 0 else 1
        return class_image, label_clf, label

if __name__ == "__main__":
    pass