python yolov7/Train.py --cfg ../configurations/yolov7_configurations/1_class/yolov7.yaml --data ../datasets/yolo_dataset_kolbottnar/data.yaml --epochs 1
python yolov7/Train.py --cfg configurations/yolov7_configurations/1_class/yolov7.yaml --data datasets/yolo_dataset_kolbottnar/data.yaml --epochs 1
python yolov7/Train.py --cfg configurations/yolov7_configurations/1_class/yolov7.yaml --data datasets/yolo_dataset_kolbottnar/data.yaml --epochs 1 --weights '' --mlflow --mlflow_path mlruns --device cpu
python yolov7/Train.py --cfg configurations/yolov7_configurations/1_class/yolov7.yaml --data datasets/yolo_dataset_kolbottnar/data.yaml --epochs 300 --weights '' --mlflow --device 0 --batch-size 12 --img-size 128 128 --hyp yolov7_configurations/1_class/hyp.scratch.p5_hillshade.yaml
python yolov7/Train.py --cfg configurations/yolov7_configurations/1_class/yolov7-tiny_small_objects.yaml --data datasets/yolo_dataset_kolbottnar/data.yaml --epochs 1 --weights '' --device cpu --batch-size 12 --img-size 128 128 --hyp yolov7_configurations/1_class/hyp.scratch.p5_hillshade.yaml
python yolov7/Train.py --cfg configurations/yolov7_configurations/1_class/yolov7-tiny_small_objects.yaml --data datasets/yolo_dataset_kolbottnar/data.yaml --epochs 1 --weights '' --device cpu --batch-size 2 --img-size 128 128 --hyp yolov7_configurations/1_class/hyp.scratch.p5_hillshade.yaml
python Train.py --cfg configurations/yolov7_configurations/1_class/yolov7-tiny_small_objects.yaml --data dataset/data_super_small.yaml --epochs 1 --weights '' --device cpu --batch-size 2 --img-size 128 128 --hyp yolov7_configurations/1_class/hyp.scratch.p5_hillshade.yaml --mlflow
python Train.py --cfg configurations/yolov7_configurations/1_class/yolov7-tiny_small_objects.yaml --data datasets/yolo_dataset_kolbottnar/data.yaml --epochs 1 --weights '' --device cpu --batch-size 2 --img-size 128 128 --hyp yolov7_configurations/1_class/hyp.scratch.p5_hillshade.yaml --mlflow
python Train.py --cfg configurations/yolov7_configurations/1_class/yolov7-tiny_small_objects.yaml --data yolo_dataset_fangstgropar_semi_auto_annotated_round_2/data.yaml --epochs 2 --weights ./checkpoints/yolov7_training.pt --device 0 --batch-size 2 --img-size 128 128 --hyp yolov7_configurations/1_class/hyp.scratch.p5_hillshade.yaml

python Train.py --cfg configurations/yolov7_configurations/1_class/yolov7-tiny_small_objects.yaml --data dataset/data_super_small.yaml --epochs 1 --weights '' --device cpu --batch-size 2 --img-size 128 128 --hyp yolov7_configurations/1_class/hyp.scratch.p5_hillshade.yaml --mlflow

python Train.py --cfg configurations/yolov7_configurations/1_class/yolov7_so.yaml --data yolo_dataset_fangstgropar_semi_auto_annotated_round_2/data.yaml --epochs 2 --weights '' --device 0 --batch-size 12 --img-size 256 256 --hyp yolov7_configurations/1_class/hyp.scratch.p5_hillshade.yaml --mlflow


python Train.py --cfg configurations/yolov7_configurations/1_class/yolov7-tiny_small_objects.yaml --data datasets/test/data.yaml --epochs 1 --weights '' --device cpu --batch-size 2 --img-size 128 128 --hyp yolov7_configurations/1_class/hyp.scratch.p5_hillshade.yaml
python Train.py --cfg configurations/yolov7_configurations/1_class/yolov7-tiny_small_objects.yaml --data yolo_dataset_fangstgropar_semi_auto_annotated_round_2/data.yaml --epochs 1 --weights '' --device cpu --batch-size 2 --img-size 128 128 --hyp yolov7_configurations/1_class/hyp.scratch.p5_hillshade.yaml


#skelettskog
python Train.py --cfg configurations/yolov7_configurations/1_class/yolov7_so.yaml --data yolo_dataset_skelettskog/data.yaml --epochs 300 --weights '' --device 0 --batch-size 8 --img-size 512 512 --hyp yolov7_configurations/1_class/hyp.scratch.p5_hillshade.yaml --mlflow --experiment_name skelettskogar

