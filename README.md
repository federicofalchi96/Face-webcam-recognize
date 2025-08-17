# Face Recognition con OpenCV e face_recognition

Questo progetto utilizza **OpenCV** e la libreria **face_recognition** (basata su dlib) per riconoscere i volti dalla webcam confrontandoli con un set di immagini di riferimento.

Quando un volto noto viene riconosciuto:
- Viene salvato uno **screenshot** nella cartella `recognized/`  
- Sul video viene mostrato un rettangolo con il nome della persona  

---

## Requisiti

- Python 3.8+
- Webcam funzionante

### Librerie necessarie
Installa le dipendenze con:


pip install opencv-python face-recognition numpy

### Avvio del programma


python recognize.py
