# Color Classification Model

This project is a machine learning model that classifies images based on their predominant colors. The model is built using TensorFlow/Keras and OpenCV.

## Project Structure

E:/training_dataset/
black/
image1.jpg
image2.jpg
...
yellow/
image1.jpg
image2.jpg
...
white/
...
...
color_classification.py
README.md

markdown
Copy code

## Requirements

- Python 3.x
- NumPy
- OpenCV
- TensorFlow
- scikit-learn

You can install the required libraries using pip:

```bash
pip install numpy opencv-python tensorflow scikit-learn
Usage
Dataset Preparation:
Ensure your dataset is structured as shown in the Project Structure section. Each folder should contain images of the respective color.

Running the Script:
Execute the provided script to train the model and evaluate its performance.

bash
Copy code
python color_classification.py
Predicting New Images:
After training the model, you can use it to predict the color of new images by specifying the path to the new image in the script.
