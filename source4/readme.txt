#꽃사진 다운로드 위치 
http://download.tensorflow.org/example_images/flower_photos.tgz

#재학습 스크립트 
https://github.com/tensorflow/hub/tree/master/examples/image_retraining

#사용방법 
1. 꽃사진을 다운로드 받음 
2. 재학습 스크립트를 다운로드 받음 
3. Project 생성 
4.   workspace 폴더생성 
5.   workspace 에 꽃사진 복사(tgz는 압축풀어서 폴더형태로)
6. retrain.py 실행(아래 스크립트로) 
python retrain.py --bottleneck_dir=./workspace/bottlenecks --model_dir=./workspace/inception --output_graph=./workspace/flowers_graph.pb --output_labels=./workspace/flowers_labels.txt --image_dir ./workspace/flower_photos --how_many_training_steps 1000