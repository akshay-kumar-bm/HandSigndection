# HandSigndection

Real-time hand-gesture recognition from a webcam, using MediaPipe hand landmarks and a scikit-learn Random Forest classifier.

## What it does
You record images of your own gestures, the pipeline extracts 21 hand landmarks per image, trains a classifier on those coordinates, and then labels gestures live from the webcam. Five gestures are mapped in `inference_classifier.py`: Peace, Yoho, Like, Cool, Shoot. (`pyautogui` is imported in the inference script, but no screen actions are triggered in the code shown.)

## Pipeline
```
collect_imgs.py -> ./data/<class>/*.jpg   (5 classes x 100 images, webcam, press Q to start each class)
create_dataset.py -> data.pickle           (MediaPipe landmarks, x/y normalised by subtracting min -> 42 features)
train_classifier.py -> model.p             (RandomForestClassifier, 80/20 stratified split, prints accuracy)
inference_classifier.py                    (live webcam: landmarks -> model -> label drawn over bounding box)
```

## Tech stack
Python, OpenCV, MediaPipe, scikit-learn, NumPy, pandas, matplotlib, PyAutoGUI.

## Setup
```bash
pip install -r requirements.txt     # opencv-python==4.7.0.68, mediapipe, scikit-learn, pyautogui
python collect_imgs.py
python create_dataset.py
python train_classifier.py
python inference_classifier.py
```
Requires a webcam. `create_dataset.py` also uses pandas and matplotlib, which are not listed in `requirements.txt`.

## Structure
```
collect_imgs.py          create_dataset.py
train_classifier.py      inference_classifier.py
data.pickle  model.p     # pre-built dataset and trained model (committed)
handsigndectoerpic.jpg   # sample image
```

## Limitations / future work
- No accuracy figure is recorded in the repo (the training script prints it at run time).
- Single-hand feature vector (first 42 values used at inference); fixed 5-gesture label map.
- Raw image data is not committed, only the pickled landmark dataset.
- Wire gesture output to `pyautogui` actions (imported but unused).
