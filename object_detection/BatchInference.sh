python BatchInference.py --coordinates 417697,6286246,600368,6798267 --img_size 512 --configuration hillshade
python BatchInference.py --coordinates 417697,6286246,600368,6798267 --img_size 512 --configuration hillshade
python BatchInference.py --coordinates 417697,6286246,600368,6798267 --img_size 512 --configuration hillshade --weights runs/train/exp12/weights/init.pt

python BatchInference.py --coordinates 447000,6390000,457000,6400000 --img_size 512 --configuration hillshade --weights L:/Arbetsgrupper/AIRaster/Projekt/2022/gemensam_ai/mlflow/mlruns/871334444541230126/adc424f283de49a3ab371bbc18a503cd/artifacts/weights/epoch_074.pt --conf-thres 0.1 --save_image
python BatchInference.py --coordinates 447000,6390000,457000,6400000 --img_size 1000 --configuration hillshade --weights L:/Arbetsgrupper/AIRaster/Projekt/2022/gemensam_ai/mlflow/mlruns/871334444541230126/adc424f283de49a3ab371bbc18a503cd/artifacts/weights/epoch_149.pt --conf-thres 0.25 --save_image
python BatchInference.py --coordinates 447000,6390000,457000,6400000 --offset 128 --img_size 128 --configuration hillshade --weights L:/Arbetsgrupper/AIRaster/Projekt/2022/gemensam_ai/mlflow/mlruns/871334444541230126/adc424f283de49a3ab371bbc18a503cd/artifacts/weights/epoch_149.pt --conf-thres 0.25 --save_image
python yolov7/detect.py --weights L:/Arbetsgrupper/AIRaster/Projekt/2022/gemensam_ai/mlflow/mlruns/871334444541230126/adc424f283de49a3ab371bbc18a503cd/artifacts/weights/epoch_149.pt --conf-thres 0.25 --source temp
python BatchInference.py --coordinates 447000,6390000,457000,6400000 --offset 128 --img_size 640 --configuration hillshade --weights L:/Arbetsgrupper/AIRaster/Projekt/2022/gemensam_ai/mlflow/mlruns/871334444541230126/adc424f283de49a3ab371bbc18a503cd/artifacts/weights/epoch_149.pt --conf-thres 0.25 --save_image
python BatchInference.py --coordinates 448200,6389200,449200,6390200 --offset 128 --img_size 256 --configuration hillshade --weights L:/Arbetsgrupper/AIRaster/Projekt/2022/gemensam_ai/mlflow/mlruns/871334444541230126/940964e4c29c41f6b8c31c812f25d2be/artifacts/weights/best_211.pt --conf-thres 0.25 --save_image
python BatchInference.py --coordinates 447700,6391100,448700,6392100 --offset 128 --img_size 256 --configuration hillshade --weights L:/Arbetsgrupper/AIRaster/Projekt/2022/gemensam_ai/mlflow/mlruns/871334444541230126/940964e4c29c41f6b8c31c812f25d2be/artifacts/weights/best_211.pt --conf-thres 0.25 --save_image
python BatchInference.py --coordinates 510000,6440000,515000,6445000 --geo 128 --img_size 1024 --configuration ortofoto_2_0 --weights L:/Arbetsgrupper/AIRaster/Projekt/2022/gemensam_ai/mlflow/mlruns/871334444541230126/940964e4c29c41f6b8c31c812f25d2be/artifacts/weights/best_211.pt --conf-thres 0.25 --save_image
python BatchInference.py --coordinates 427700,6291100,498700,6492100 --geo 1280 --offset 10000 --img_size 256 --configuration hillshade --weights L:/Arbetsgrupper/AIRaster/Projekt/2022/gemensam_ai/mlflow/mlruns/871334444541230126/940964e4c29c41f6b8c31c812f25d2be/artifacts/weights/best_211.pt --conf-thres 0.25



python BatchInference.py --coordinates 430000,6370000,420000,6380000 --geo 1280 --img_size 256 --configuration hillshade --weights L:/Arbetsgrupper/AIRaster/Projekt/2022/gemensam_ai/mlflow/mlruns/871334444541230126/940964e4c29c41f6b8c31c812f25d2be/artifacts/weights/best_211.pt --conf-thres 0.25


python BatchInference.py --coordinates 430000,6370000,440000,6380000 --geo 1280 --img_size 256 --configuration hillshade --algorithm yolor --weights runs/train/fangstgropar16/weights/best.pt --conf-thres 0.25 --yolor_cfg configurations/yolor_configurations/1_class/yolor_p6_fangstgropar.cfg --device cpu
python BatchInference.py --coordinates 422408,6972464,522408,7072464 --geo 128 --img_size 128 --configuration hillshade --algorithm yolor --weights runs/best_overall.pt --conf-thres 0.5 --yolor_cfg configurations/yolor_configurations/1_class/yolor_p6_fangstgropar.cfg --device cpu