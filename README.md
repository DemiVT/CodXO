# Artificial Intelligence Intern Projects

This repository contains implementations of various artificial intelligence projects completed as part of an internship. The projects are categorized into basic (for silver certification) and intermediate (for gold certification) levels.

## Table of Contents

- [Basic Projects](#basic-projects)
  - [Fake News Detector](#fake-news-detector)
  - [Autocorrect Tool](#autocorrect-tool)
  - [Translator App](#translator-app)
- [Intermediate Projects](#intermediate-projects)
  - [Object Detection System](#object-detection-system)
  - [Hand Gesture Recognition](#hand-gesture-recognition)
  - [Food Item Recognition and Calorie Estimation](#food-item-recognition-and-calorie-estimation)

## Basic Projects

### Fake News Detector

**Objective:** Detect fake news articles using natural language processing techniques.

**Files:**
- `fake_news_data.csv` - Dataset containing news articles with labels.
- `fake_news_detector.py` - Python script implementing the fake news detection model.

**Usage:**
1. Ensure the `fake_news_data.csv` file is in the same directory as the script.
2. Run the script:
    ```bash
    python fake_news_detector.py
    ```

**Requirements:**
- `pandas`
- `scikit-learn`

### Autocorrect Tool

**Objective:** Implement an autocorrect tool to suggest corrections for misspelled words.

**Files:**
- `word_corpus.txt` - Text file containing a corpus of words for spell checking.
- `autocorrect_tool.py` - Python script for the autocorrect tool.

**Usage:**
1. Ensure the `word_corpus.txt` file is in the same directory as the script.
2. Run the script:
    ```bash
    python autocorrect_tool.py
    ```

**Requirements:**
- `nltk`

### Translator App

**Objective:** Create a simple translator app to translate text between different languages.

**Files:**
- `translator.py` - Python script implementing the translator app using translation APIs.

**Usage:**
1. Install the `googletrans` library if not already installed:
    ```bash
    pip install googletrans==4.0.0-rc1
    ```
2. Run the script:
    ```bash
    python translator.py
    ```

**Requirements:**
- `googletrans`

## Intermediate Projects

### Object Detection System

**Objective:** Implement an object detection system using deep learning techniques.

**Files:**
- `object_detection_data/` - Directory containing labeled images for object detection.
- `object_detection.py` - Python script implementing the object detection model.

**Usage:**
1. Ensure the object detection dataset is in the `object_detection_data` directory.
2. Run the script:
    ```bash
    python object_detection.py
    ```

**Requirements:**
- `tensorflow`
- `opencv-python`

### Hand Gesture Recognition

**Objective:** Develop a hand gesture recognition model to identify and classify different hand gestures.

**Files:**
- `hand_gesture_data/` - Directory containing labeled images of hand gestures.
- `hand_gesture_recognition.py` - Python script for gesture recognition.

**Usage:**
1. Ensure the hand gesture data is in the `hand_gesture_data` directory.
2. Ensure the pre-trained model file `hand_gesture_model.h5` is in the same directory as the script.
3. Run the script:
    ```bash
    python hand_gesture_recognition.py
    ```

**Requirements:**
- `tensorflow`
- `opencv-python`

### Food Item Recognition and Calorie Estimation

**Objective:** Develop a model to recognize food items from images and estimate their calorie content.

**Files:**
- `food_images/` - Directory containing images of food items.
- `food_recognition.py` - Python script for food recognition.

**Usage:**
1. Ensure the food images are in the `food_images` directory.
2. Run the script:
    ```bash
    python food_recognition.py
    ```

**Requirements:**
- `tensorflow`
- `keras`
- `opencv-python`

## Installation and Setup

To run the provided scripts, you'll need to install the following Python libraries:

```bash
pip install pandas scikit-learn nltk googletrans tensorflow opencv-python
