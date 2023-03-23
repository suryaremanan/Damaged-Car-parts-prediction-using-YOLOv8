# Damaged-Car-parts-prediction-using-YOLOv8

## Training

``` pip install ultralytics```

* Collect dataset of damaged cars
* Annotate them
* in this case there are 8 classes namely : damaged door, damaged window, damaged headlight, damaged mirror, dent, damaged hood, damaged bumper, damaged windshield
* convert the annotations into YOLO 1.1 format
* run ```main.py```
* in this particular case i ran for 5000 epochs



![Screenshot from 2023-03-18 19-50-29](https://user-images.githubusercontent.com/17148269/226126564-145ebe9f-e276-44f9-b4cd-023843aafd1a.png)


## Results





![F1_curve](https://user-images.githubusercontent.com/17148269/226126757-21ec1c85-32a6-4e6f-ab51-6d07bf37c2e7.png)
![P_curve](https://user-images.githubusercontent.com/17148269/226126770-2d03e247-53df-4e82-9b1c-612126319ac1.png)
![PR_curve](https://user-images.githubusercontent.com/17148269/226126832-83ee056e-8711-4f0c-9c80-b23810aa42e0.png)
![R_curve](https://user-images.githubusercontent.com/17148269/226126876-132571a1-45b1-4595-bf1e-b30425f4b73f.png)



## Prediction
![val_batch0_pred](https://user-images.githubusercontent.com/17148269/226127656-dce9f1ed-fc2d-41be-8400-678b6bbc0cb3.jpg)
![val_batch1_pred](https://user-images.githubusercontent.com/17148269/226127705-c9fddde3-b5cf-4266-91cf-08d5705aa452.jpg)
![val_batch2_pred](https://user-images.githubusercontent.com/17148269/226127766-4bb44851-a3af-4c06-a86c-c0fdbd3803f8.jpg)
## Deployment

![Screenshot from 2023-03-20 06-11-43](https://user-images.githubusercontent.com/17148269/226222000-41ed7812-b1d2-4d4f-a5ee-97230d0eba98.png)

## Usage

* Clone the repo
* cd to my_fastapi_app
* run ```python main.py```

Test with images
Done!
