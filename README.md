# Student Placement Prediction System

A machine learning-based web application that predicts whether a student is likely to be placed based on academic performance, technical skills, internships, projects, and other relevant factors.

## Features

- Predicts student placement status
- Machine Learning model trained on placement dataset
- Responsive HTML/CSS user interface
- Flask backend
- Docker support
- Ready for deployment on Render

## Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas
- HTML
- CSS
- Docker
- Render

## Project Structure

```
placement-predictor/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
│
├── dataset/
│   └── placement_data.csv
│
├── model/
│   ├── train_model.py
│   └── placement_model.pkl
│
├── static/
│   └── style.css
│
└── templates/
    └── index.html
```

## Installation

1. Clone the repository

```bash
git clone <https://github.com/gunal777/placement_prediction>
cd placement-predictor
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the application

```bash
python app.py
```

4. Open your browser

```
http://127.0.0.1:5000
```

## Docker

Build the Docker image

```bash
docker build -t placement_predictor .
```

Run the Docker container

```bash
docker run -p 5000:5000 placement_predictor
```

Then open:

```
http://localhost:5000
```

## Machine Learning

The project compares multiple machine learning algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

The best-performing model is automatically saved as:

```
model/placement_model.pkl
```