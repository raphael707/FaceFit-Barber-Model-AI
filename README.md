# FaceFit Barber Model AI - Face Shape Classification Model for Hairstyle Recommendation

"FaceFit Barber Model AI" is the machine learning repository for the FaceFit Barber capstone project. This repository contains the process of building, training, evaluating, and exporting an AI model that classifies user face shapes to support personalized hairstyle recommendations.

The model is designed to analyze a user's face image and classify it into one of several face shape categories. The prediction result is then used by the FaceFit Barber web application to recommend suitable haircut styles based on the detected face shape.

## Main Features

* 🧠 **Face Shape Classification**: Classifies user face images into predefined face shape categories.
* 🖼️ **Image Preprocessing**: Applies image preprocessing before training and inference.
* 📊 **Model Evaluation**: Provides evaluation results such as confusion matrix, training curves, model architecture, and training report.
* 💾 **Exported AI Model**: Saves the trained model in `.keras` and TensorFlow SavedModel formats.
* 📁 **Training Artifacts**: Stores model reports, class names, preprocessing configuration, and architecture visualization.
* 🚀 **Deployment Ready**: The exported `.keras` model can be integrated into the FastAPI AI backend service.

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pillow
* Matplotlib
* Google Colab
* Jupyter Notebook

## Project Structure

```text
FaceFit-Barber-Model-AI/
├── configs/
│   ├── class_names.json
│   └── preprocessing_config.json
├── models/
│   ├── saved_model/
│   ├── best_face_shape_model.keras
│   └── face_shape_model.keras
├── notebooks/
│   └── face_shape_detection.ipynb
├── reports/
│   ├── confusion_matrix.png
│   ├── model_architecture.png
│   ├── training_curves.png
│   └── training_report.json
└── README.md
```

## Model Objective

The main objective of this model is to classify a user's face shape based on an input face image. The predicted face shape becomes the main reference for the hairstyle recommendation system in the FaceFit Barber application.

The model helps the system identify whether the user's face shape belongs to one of the supported categories. After the prediction is generated, the application maps the detected face shape to suitable haircut recommendations.

## Face Shape Classes

The model classifies face images into the following categories:

```text
diamond
heart
oval
round
square
```

These classes are stored in:

```text
configs/class_names.json
```

The class order is important because the backend inference process must use the same class order as the model training process.

## Input Specification

The model receives RGB face images with the following input shape:

```text
224 x 224 x 3
```

This means each input image is resized to 224 pixels in width, 224 pixels in height, and contains 3 color channels: Red, Green, and Blue.

## Image Preprocessing

Before being used by the model, each image goes through several preprocessing steps:

1. Convert the image to RGB format.
2. Apply center crop with a crop ratio of 0.88.
3. Resize the image to 224 x 224 pixels.
4. Convert the image into a `float32` array.
5. Pass the processed image into the model.

The image is not manually divided by `/255` in the backend because the model already includes a `Rescaling` layer inside the architecture.

The preprocessing configuration is stored in:

```text
configs/preprocessing_config.json
```

## Model Output

The model outputs probability scores for each face shape class. The class with the highest probability is selected as the final predicted face shape.

Example output:

```json
{
  "face_shape": "heart",
  "confidence": 0.637,
  "probabilities": {
    "diamond": 0.257,
    "heart": 0.637,
    "oval": 0.025,
    "round": 0.080,
    "square": 0.000
  }
}
```

## Training Process

The training process is documented in the notebook located in the `notebooks/` folder. The notebook contains the full model development workflow, starting from dataset preparation, preprocessing, model building, training, evaluation, and model export.

General training workflow:

```text
Dataset preparation
→ Image preprocessing
→ Model architecture building
→ Model training
→ Model evaluation
→ Model export
```

The main notebook file:

```text
notebooks/face_shape_detection.ipynb
```

## Evaluation Results

The model evaluation files are stored in the `reports/` and `configs/` folders.

* `reports/confusion_matrix.png` shows the classification performance for each face shape class.
* `reports/training_curves.png` shows the training and validation performance during training.
* `reports/model_architecture.png` shows the architecture of the trained model.
* `configs/training_report.json` stores detailed training and evaluation information.

These files are used to analyze the performance of the model and understand how well the model performs on each face shape category.

## Exported Model

The trained model is exported into several formats:

```text
models/face_shape_model.keras
models/best_face_shape_model.keras
models/saved_model/
```

The `.keras` model is used for integration with the FaceFit Barber AI backend service. The SavedModel format is also stored as an additional export format.

## Configuration Files

The `configs/` folder contains supporting configuration files:

* `class_names.json` stores the class labels used by the model.
* `preprocessing_config.json` stores preprocessing information such as image size and crop configuration.
* `training_report.json` stores training and evaluation information.

These files help ensure that the backend inference process uses the same class order and preprocessing settings as the training process.

## Relationship with FaceFit Barber Application

This repository focuses only on the AI model development process. The exported model from this repository is used by the FaceFit Barber backend AI service to perform face shape prediction.

Application flow:

```text
User uploads or captures face image
→ AI backend preprocesses the image
→ Model predicts face shape
→ Backend returns prediction result
→ Frontend displays hairstyle recommendation
```

The model output is used as the basis for hairstyle recommendation in the FaceFit Barber web application.

## Deployment Usage

The exported `.keras` model is integrated into the FaceFit Barber AI backend using FastAPI and deployed on Hugging Face Spaces.

Production AI Backend:

```text
https://justblaisee-facefit-ml-service.hf.space
```

Main endpoint used by the frontend:

```text
https://justblaisee-facefit-ml-service.hf.space/api/faces/analyze
```

## Repository Scope

This repository is specifically used for AI model development and training artifacts. It does not contain the full frontend or backend application source code.

The FaceFit Barber project is separated into different repositories based on responsibility:

```text
Frontend Repository
→ Web interface, camera integration, UI, and user interaction

Backend Repository
→ API routing, local backend integration, and ML service integration

Model AI Repository
→ Dataset processing, training notebook, model evaluation, and exported model files
```

## Notes

* This repository is focused on model development and training artifacts.
* Dataset files are not included if the dataset size is too large.
* Model files such as `.keras` may require Git LFS if they exceed GitHub file size limits.
* The model output is used as the basis for hairstyle recommendation in the FaceFit Barber application.
* The backend service and frontend application are stored in separate repositories.
* The preprocessing process used during inference should match the preprocessing process used during training.
* The class order in `class_names.json` must match the model output order.

## Contributing

Feel free to fork this repository and submit a pull request.

## License

MIT License - Use freely with attribution.

## Credits

Developed as part of the **FaceFit Barber Capstone Project** to support AI-based face shape analysis and personalized hairstyle recommendation.
