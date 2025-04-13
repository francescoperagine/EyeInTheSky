# Eye in the Sky: Drone-based Object Detection

## Overview

"Eye in the Sky" is a computer vision project developed as a thesis project for the Master of Science degree in Computer Science with Artificial Intelligence specialization at the Università degli Studi di Bari (UniBA). This research was supervised by Professor Gennaro Vessio and PhD Dr. Pasquale De Marinis.

Focused on optimizing object detection models for drone imagery, it uses [YOLOv12](https://github.com/sunsmarterjie/yolov12) models within [Ultralytics](https://www.ultralytics.com/)  package and implements knowledge distillation techniques to create efficient, lightweight models suitable for real-time detection on resource-constrained devices.

The system can detect multiple object categories from the VisDrone dataset, including people, vehicles, and other common road/urban elements. While the current implementation focuses on object detection fundamentals, it provides the foundation for more advanced applications like traffic monitoring, search and rescue, or emergency response assistance.

## Key Features

- **Model Distillation**: Transfer knowledge from larger, more accurate teacher models to smaller, faster student models
- **Custom Dataset Handling**: Modified VisDrone dataset with class merging for improved performance
- **Feature Adaptation**: Innovative feature adaptation layers to maximize knowledge transfer
- **Training Visualization**: Comprehensive metrics tracking and visualization
- **Performance Optimization**: Cyclical learning rates and adaptive loss weighting

## Project Structure

This project follows the [Cookiecutter Data Science v2](https://github.com/drivendata/cookiecutter-data-science) template:

```
├── eyeinthesky        <- Source code package
│   ├── notebooks      <- Jupyter notebooks for model training and distillation
│   ├── modeling       <- Scripts for model training and evaluation
├── requirements.txt   <- Package dependencies
├── setup.py           <- Package installation script
├── .env-sample        <- Sample environment variables template
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Weights & Biases account
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/francescoperagine/EyeInTheSky.git
cd eyeinthesky
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package and dependencies:
```bash
pip install -e .
pip install -r requirements.txt
```

4. Set up Weights & Biases:
   - Create a `.env` file in the root directory using `.env-sample` as a template
   - Add your W&B API key to the `.env` file

## Usage

### Using Jupyter Notebooks

For experimentation and development:

1. Navigate to the notebooks directory:
```bash
cd eyeinthesky/notebooks
```
2. Prepare the dataset once with `prepare_dataset.ipynb`
3. Train a teacher model using `train.ipynb`
4. Run distillation using `distillation.ipynb`

### Key Components

The distillation process includes several innovative components:

- **FeatureAdaptation**: Custom layers that adapt features from the student model to match the teacher
- **DecayingCyclicalLR**: Learning rate scheduler with adaptive decay for optimal convergence
- **DistillationCallback**: Core component managing the knowledge transfer process
- **VisDroneDataset**: Custom dataset handler with class merging (pedestrian + people → person)

## Technical Details

### Model Architecture

The project uses YOLOv12 models at different scales:
- Teacher model: YOLOv12x (larger, more accurate)
- Student model: YOLOv12n (smaller, faster)

### Distillation Approach

The knowledge distillation process focuses on feature-level knowledge transfer:
1. **Feature-level distillation**: Transferring intermediate feature representations from specific network layers
2. **Feature adaptation:**: Custom neural network layers transform student features to match teacher dimensions
3. **Cosine similarity loss**: Measuring and minimizing the difference between adapted student features and teacher features
4. **Adaptive loss weighting**: Dynamically balancing the contribution of detection loss vs. distillation loss using learnable parameters
5. **Layer importance weighting**: Learning the relative importance of different layers in the distillation process

## Project Roadmap

Future development directions may include:
- Expanding to different drone altitudes and perspectives
- Furtherly customize the VisDrone dataset to better adapt it to the local context
- Real-time deployment on edge devices
- Integration with streamed video analysis
- Adding semantic segmentation capabilities

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/francescoperagine/EyeInTheSky/blob/main/LICENSE) file for details.