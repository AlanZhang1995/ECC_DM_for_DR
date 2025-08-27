docker run --name ECC_DM --gpus all -it --shm-size=16g -v /home/haochen/Documents:/home/haochen/Documents -v /media/haochen/WD8TB:/home/haochen/DataComplex pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

<<COMMENT_DM_playground
# install addition packeges for pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
pip install opencv-python-headless pandas wandb torchnet scikit-learn
pip install accelerate==0.26.1
pip install diffusers==0.27.2
pip install transformers==4.37.2
pip install peft==0.7.0
pip install huggingface-hub==0.20.3
pip install torch==2.2.2
pip install torchvision==0.17.2
COMMENT_DM_playground