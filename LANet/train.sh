# test
# python main_base.py --model res50 --visname tests --n_classes 5 --test True
# python main_lanet.py --model inceptionv3 --adaloss True --visname tests --test True

#python main_base.py --n_classes 5 --model inceptionv3 --visname ddr_inceptionv3_base5
#python main_lanet.py --model inceptionv3 --visname ddr_inceptionv3_lanet
#python main_lanet.py --model inceptionv3 --adaloss True --visname ddr_inceptionv3_lanet_adl --epochs 300
#python main_lanet.py --model inceptionv3 --adaloss True --visname ddr_inceptionv3_lanet_adl_seed2 --epochs 300 --seed 2

#python main_base.py --n_classes 5 --model vgg --visname ddr_vgg_base5 -bs 26
#python main_lanet.py --model vgg --visname ddr_vgg_lanet -bs 26
#python main_lanet.py --model vgg --adaloss True --visname ddr_vgg_lanet_adl -bs 26

#python main_base.py --n_classes 5 --model dense121 --visname ddr_dense121_base5  -bs 30
#python main_lanet.py --model dense121 --visname ddr_dense121_lanet  -bs 30
#python main_lanet.py --model dense121 --adaloss True --visname ddr_dense121_lanet_adl --epochs 200  -bs 30

#python main_lanet.py --model vgg --adaloss True --visname ddr_vgg_lanet_adl_f0_os -bs 26 --fold 0 --use_sampler

#python main_lanet.py --model dense121 --adaloss True --visname ddr_dense121_lanet_adl_f0_os --epochs 200  -bs 30 --fold 0 --use_sampler
#python main_lanet.py --model inceptionv3 --adaloss True --visname ddr_inceptionv3_lanet_adl_f1_os --epochs 300 --fold 1 --use_sampler


#CUDA_VISIBLE_DEVICES=0 python main_lanet.py --model dense121 --adaloss True --visname ddr_dense121_lanet_adl_f0_os_sync_o_acc_seed186_lr-5 -bs 30 --epochs 200 --seed 186 --fold 0 --use_sampler --sync_data w/o --lr 0.00001
#CUDA_VISIBLE_DEVICES=0 python main_lanet.py --model dense121 --adaloss True --visname ddr_dense121_lanet_adl_f1_os_sync_o_acc_seed186_lr-5 -bs 30 --epochs 200 --seed 186 --fold 1 --use_sampler --sync_data w/o --lr 0.00001
#CUDA_VISIBLE_DEVICES=0 python main_lanet.py --model dense121 --adaloss True --visname ddr_dense121_lanet_adl_f2_os_sync_o_acc_seed186_lr-5 -bs 30 --epochs 200 --seed 186 --fold 2 --use_sampler --sync_data w/o --lr 0.00001
#CUDA_VISIBLE_DEVICES=0 python main_lanet.py --model dense121 --adaloss True --visname ddr_dense121_lanet_adl_f3_os_sync_o_acc_seed186_lr-5 -bs 30 --epochs 200 --seed 186 --fold 3 --use_sampler --sync_data w/o --lr 0.00001
#CUDA_VISIBLE_DEVICES=0 python main_lanet.py --model dense121 --adaloss True --visname ddr_dense121_lanet_adl_f4_os_sync_o_acc_seed186_lr-5 -bs 30 --epochs 200 --seed 186 --fold 4 --use_sampler --sync_data w/o --lr 0.00001

CUDA_VISIBLE_DEVICES=0 python main_lanet.py --model vgg --adaloss True --visname ddr_vgg_lanet_adl_f0_os_sync_wo_acc -bs 26 --fold 0 --use_sampler --sync_data w/o
CUDA_VISIBLE_DEVICES=0 python main_lanet.py --model vgg --adaloss True --visname ddr_vgg_lanet_adl_f1_os_sync_wo_acc -bs 26 --fold 1 --use_sampler --sync_data w/o
#CUDA_VISIBLE_DEVICES=0 python main_lanet.py --model vgg --adaloss True --visname ddr_vgg_lanet_adl_f2_os_sync_wo_acc -bs 26 --fold 2 --use_sampler --sync_data w/o
#CUDA_VISIBLE_DEVICES=0 python main_lanet.py --model vgg --adaloss True --visname ddr_vgg_lanet_adl_f3_os_sync_wo_acc -bs 26 --fold 3 --use_sampler --sync_data w/o
#CUDA_VISIBLE_DEVICES=0 python main_lanet.py --model vgg --adaloss True --visname ddr_vgg_lanet_adl_f4_os_sync_wo_acc -bs 26 --fold 4 --use_sampler --sync_data w/o
