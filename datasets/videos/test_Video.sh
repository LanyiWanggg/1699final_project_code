#model_path=snapshots/300W-CPM-DET-500Epoch/checkpoint/cpm_vgg16-epoch-499-500.pth

# model_path=snapshots_old/300W-CPM-DET-500Epoch/checkpoint/cpm_vgg16-epoch-499-500.pth
# python ./demo_vid.py --model $model_path --video datasets/videos/Lanyi.mp4 --post_str CPM_VGG_500

# model_path=snapshots/300W-CPM-DET-200Epoch/checkpoint/cpm-epoch-199-200.pth
# python ./demo_vid.py --model $model_path --video datasets/videos/Lanyi.mp4 --post_str CPM_200
# python ./demo_vid.py --model $model_path --video datasets/videos/Lanyi.mp4 --post_str CPM_200 --temporal
# python ./demo_vid.py --model $model_path --video datasets/videos/Liling_1.mp4 --post_str CPM_200
# python ./demo_vid.py --model $model_path --video datasets/videos/Liling_1.mp4 --post_str CPM_200 --temporal
# python ./demo_vid.py --model $model_path --video datasets/videos/Liling_2.mp4 --post_str CPM_200
# python ./demo_vid.py --model $model_path --video datasets/videos/Liling_2.mp4 --post_str CPM_200 --temporal

model_path=snapshots/300W-CPM-VGG-DET-200Epoch/checkpoint/cpm_vgg16-epoch-199-200.pth
python ./demo_vid.py --model $model_path --video datasets/videos/1.mp4 --post_str CPM_VGG_200 --temporal
python ./demo_vid.py --model $model_path --video datasets/videos/Lanyi.mp4 --post_str CPM_VGG_200 --temporal
python ./demo_vid.py --model $model_path --video datasets/videos/Liling_1.mp4 --post_str CPM_VGG_200 --temporal
python ./demo_vid.py --model $model_path --video datasets/videos/Liling_2.mp4 --post_str CPM_VGG_200 --temporal