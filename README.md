# Face and Body Detection using OpenCV

This Python script utilizes OpenCV to perform real-time detection of frontal faces, side faces, and full bodies using a webcam. The program captures video frames, processes them to detect various features, and displays the results with visual annotations.

## Features

- **Real-time face and body detection**: Detects frontal faces, side faces, and full bodies using Haar Cascade classifiers.
- **Visual annotations**: Draws rectangles, lines, and text around detected faces and bodies.
- **Midpoint calculation**: Computes and displays the midpoint of detected frontal faces.
- **Dynamic updates**: Continuously processes the video stream and updates the display in real-time.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Rcode879/Haar-cascade-detector.git
    cd face-body-detection
    ```

2. **Install the required libraries**:

    Ensure you have Python installed (preferably Python 3.6 or higher) and install the required dependencies:

    ```bash
    pip install opencv-python
    ```

## Usage

1. **Run the script**:

    Execute the script to start the webcam and begin detecting faces and bodies:

    ```bash
    python detector.py
    ```

2. **Interact with the application**:

    - The webcam feed will open in a window named `frame`.
    - The program will highlight detected faces and bodies with annotations.
    - To stop the program, press the `q` key.

## How It Works

- **Capture Video**: Initializes the video capture from the default webcam.
- **Set Resolution**: Configures the capture to HD resolution (1280x720).
- **Load Haar Cascades**: Loads pre-trained Haar Cascade classifiers for detecting frontal faces, side faces, and full bodies.
- **Process Each Frame**:
  - Converts each frame to grayscale for efficient detection.
  - Uses the Haar Cascade classifiers to detect frontal faces, side faces, and bodies.
  - Draws rectangles, lines, and text annotations around the detected objects.
  - Displays the annotated frame in a window.
- **Exit on Key Press**: Ends the loop and releases resources when the `q` key is pressed.

## Dependencies

- [OpenCV](https://opencv.org/) (cv2)

## Acknowledgements

This project utilizes the OpenCV library for image processing and computer vision tasks. Haar Cascade classifiers are provided by OpenCV's pre-trained models.


