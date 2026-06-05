# Wi-Fi-Based-Real-Time-Through-Wall-Human-Activity-Recognition-Architecture
This Final Year Project presents a real-time Human Activity Recognition (HAR) and Occupancy Detection system based on Frequency-Modulated Continuous Wave (FMCW) radar technology. The system leverages radar signal processing and machine learning techniques to recognize human activities without requiring wearable devices or cameras, ensuring privacy-preserving monitoring.

Project Highlights:
1) Developed a complete FMCW radar-based sensing framework using custom-collected radar data.
2) Created and labeled our own dataset for all experiments and model training.
3) Achieved 95% activity recognition accuracy for five indoor activities(walking,running,sitting,standing,falling) within a range of 2 meters in real time sensing.
4) Successfully recognized three human activities(walking,running,falling) through walls at distances of 4–5 meters, achieving 88% accuracy real time.
5) Implemented real-time occupancy detection to determine the presence or absence of individuals within the monitored area.
6) Designed a complete pipeline including radar data acquisition, signal preprocessing, feature extraction, dataset generation, model training, and real-time inference.

Technologies Used:

1) FMCW Radar
2) USRP (Universal Software Radio Peripheral)
3) MATLAB
4) Python
5) Signal Processing
6) Machine Learning / Deep Learning
7) Real-Time Activity Classification

## System Pipeline

The system operates in real time by continuously transmitting and receiving FMCW radar signals using a USRP-based radar setup. The received signals are processed to generate range information and extract motion-related characteristics of targets within the sensing area.

The processed radar data undergoes signal conditioning and feature extraction to capture patterns associated with different human activities. These features are then passed to trained machine learning models that perform occupancy detection and activity classification.

The system first determines whether a person is present in the monitored environment. If occupancy is detected, the activity recognition module classifies the observed motion into predefined activities. The framework supports both in-room sensing and through-wall sensing, enabling activity recognition even when obstacles block direct line-of-sight.

The complete pipeline—from radar signal acquisition, preprocessing, and feature extraction to machine learning inference—operates in real time, providing continuous monitoring and immediate activity predictions while preserving user privacy.
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/1a02ec91-c6c5-403a-a9f1-0f1f240a1feb" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/fe45a568-2775-4c40-8b04-8b37eb8c75a0" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/cfaa0a39-d1f6-445e-be81-888964dbf985" />


