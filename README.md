# Machine Learning Engineer Test: Computer Vision and Object Detection

## Objective
This test aims to assess your skills in computer vision and object detection, with a specific focus on detecting room walls and identifying rooms in architectural blueprints or pre-construction plans.

This test evaluates your practical skills in applying advanced computer vision techniques to a specialized domain and your ability to integrate machine learning models into a simple API server for real-world applications.

Choose one of the visual tasks, one of the text extraction tasks, and the API Server task. We encourage you to submit your tests even if you can’t complete all tasks.

Good luck!


## Full test description
[Senior Machine Learning Engineer.pdf](https://github.com/user-attachments/files/16702909/Senior.Machine.Learning.Engineer.pdf)

## Full test Solution Document
[ml-eng-test-solution.pdf](https://github.com/user-attachments/files/16702909/Senior.Machine.Learning.Engineer.pdf)


## PS
Share your project with the following GitHub users:
- vhaine-tb
- gabrielreis-tb

1. Clone the repository:
```
git clone https://github.com/ammarfitwalla/ml-eng-test.git
```
2. Change the directory
```
cd ml-eng-test
```
3. Create a Directory
```
mkdir CubiCasa
```
4. Download the trained model:  
Due to size constraints, the trained model file is not included in this repository. Since this repository has been forked from another repository, Git LFS (Large File Storage) cannot be used to add larger files. You need to manually download the model using the following link:
    - [Download the trained CubiCasa model](https://drive.google.com/file/d/1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK/view)

Once downloaded, place the model file in the `CubiCasa` directory.

4. Build the Docker container:
```
docker build -t docker-group .
```
5. Run the Docker container:
```
sudo docker run -d -p 3000:3000 docker-group
```
## Usage

```
curl -X POST "http://127.0.0.1:3000/predict/detect_wall" -H "Content-Type: multipart/form-data" -F "file=@test_wall.png" --output output_walls.png
curl -X POST "http://127.0.0.1:3000/predict/detect_wall" -H "Content-Type: multipart/form-data" -F "file=@test_wall_2.png" --output output_walls_2.png
curl -X POST "http://127.0.0.1:3000/predict/room" -H "Content-Type: multipart/form-data" -F "file=@F1_scaled.png" --output output_room.png
curl -X POST "http://127.0.0.1:3000/predict/icon" -H "Content-Type: multipart/form-data" -F "file=@F1_scaled.png" --output output_icon.png

```
