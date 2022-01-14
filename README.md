# YOLO-Api
Object detection webapp using YOLOv3

## Usage
```bash
conda env create -f environment.yml
conda activate OD-webapp
```

## Download YOLOv3 pretrained weights
```
wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights
```

## Model loading
The following step will convert the yolov3 weights into tensorflow .ckpt extension
```
python load_weights.py
```

## Run the flask app
```bash
python app.py
```

## Test the API
```bash
curl -X POST -F images=@data/images/dog.jpg "http://localhost:5000/detections"
```

## Raw testing (support for yolov3 tiny included)
```bash
# yolov3
python detect.py --images "data/images/dog.jpg, data/images/office.jpg"

# yolov3-tiny
python detect.py --weights ./weights/yolov3-tiny.tf --tiny --images "data/images/dog.jpg"
```