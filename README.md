# 🤟 ASL Hand Sign Recognition

A **Streamlit-based web application** that recognizes American Sign Language (ASL) hand signs from images or webcam input using a pre-trained **TensorFlow/Keras model**.

---

## **Features**

- Upload an image of a hand sign for prediction.  
- Use your webcam to capture hand signs directly.  
- Supports **29 classes**: A–Z, delete (`del`), nothing, and space.  
- Confidence threshold ensures only accurate predictions are shown.  
- Lightweight, easy-to-use web interface with Streamlit.  

---

## **Model**

- The model file `asl_model.h5` is tracked using **Git LFS** due to its size.  
- Input images are resized to **128x128 pixels** and normalized before prediction.  
- Uses `TensorFlow/Keras` for inference.  

---

## **Requirements**

- Python 3.9+  
- Streamlit  
- TensorFlow  
- Pillow  
- NumPy  
- Git LFS (for handling the `.h5` model file)

Install dependencies via:

```bash
pip install streamlit tensorflow pillow numpy git-lfs
