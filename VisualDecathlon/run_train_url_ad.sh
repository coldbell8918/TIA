# Train the URL model with ResNet-26 with parallel adapters
CUDA_VISIBLE_DEVICES=0 python train_url_ad.py --dataset imagenet12 aircraft cifar100 daimlerpedcls dtd gtsrb vgg-flowers omniglot svhn ucf101 --wd 1. --lr 0.01 --alr 1e-2 --mode parallel_adapters --expdir ./results/url_ad/ --datadir ./data/decathlon-1.0/ --imdbdir ./data/decathlon-1.0/annotations/ --sdl-root ./results/sdl --adaptor-opt linear --sigma 1 --beta 1 --source ./data/sdl/checkpoint/ckptbnimagenet12_best.pth
