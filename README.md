---
title: FreshVegFruit
emoji: 🐠
colorFrom: pink
colorTo: red
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# FreshVegFruit 🍎🍅

## Overview
FreshVegFruit is a project developed for Flipkart GRID 6.0, aiming to leverage OCR and image recognition to enhance product packaging analysis and freshness detection. The project involves using machine learning models to recognize brands, count product quantities, and assess the freshness of fruits and vegetables.

## Features
- **OCR to Extract Details**: Use OCR to read brand details, pack size, brand name, etc.
- **Expiry Date Validation**: Validate expiry dates using OCR to read expiry and MRP details printed on items.
- **Image Recognition**: Recognize brands and count product quantities from images.
- **Freshness Detection**: Assess the freshness of fruits and vegetables by analyzing various visual cues and patterns.

## Team Members
- **Abhishek Kumar Choudhary** <img src="myimage.jpg" width="100" height="100">
- **Aman Deep** <img src="aman.jpg" width="100" height="100">
- **Gaurav Lodhi** <img src="gaurav.jpg" width="100" height="100">
- **Anand Jha** <img src="anandimg.jpg" width="100" height="100">

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/meakc/FreshVegFruit.git
    cd FreshVegFruit
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Navigate to the provided URL to access the application.

## Project Structure
- `app.py`: Main application file.
- `requirements.txt`: List of dependencies.
- `README.md`: Project documentation.
- `images/`: Directory containing images used in the project.

## Configuration
The project uses the following configuration:
```yaml
---
title: FreshVegFruit
emoji: 🐠
colorFrom: pink
colorTo: red
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
---
```
