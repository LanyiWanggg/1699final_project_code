#model_path=snapshots/300W-CPM-DET/checkpoint/cpm_vgg16-epoch-049-050.pth
#model_path=snapshots/300W-CPM-DET-200Epoch/checkpoint/cpm-epoch-199-200.pth
model_path=snapshots/300W-CPM-VGG-DET-200Epoch/checkpoint/cpm_vgg16-epoch-199-200.pth

python ./demo.py --model $model_path --image datasets/images/image_0019.png
python ./demo.py --model $model_path --image datasets/images/image_0020.png
python ./demo.py --model $model_path --image datasets/images/image_0028.png
python ./demo.py --model $model_path --image datasets/images/Liling_7.png
python ./demo.py --model $model_path --image datasets/images/Liling_1.png
python ./demo.py --model $model_path --image datasets/images/Liling_2.png
python ./demo.py --model $model_path --image datasets/images/Liling_3.png
python ./demo.py --model $model_path --image datasets/images/Liling_4.png
python ./demo.py --model $model_path --image datasets/images/Liling_5.png
python ./demo.py --model $model_path --image datasets/images/Liling_6.png
python ./demo.py --model $model_path --image datasets/images/Liling_7.png