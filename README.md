# 🎨 Neural Style Transfer App

A Flask web application that applies **Neural Style Transfer (NST)** using the **AdaIN (Adaptive Instance Normalization)** algorithm to blend the content of one image with the artistic style of another.

---

## 🖼️ Demo

Upload a content image and a style image, adjust the style strength, and get a stylized output in seconds!

---

## 🚀 Features

- Upload any content and style image (JPG/PNG)
- Adjustable **style strength** (alpha) slider
- Real-time stylization using a pre-trained AdaIN decoder
- Download the stylized result
- Built-in example images to try

---

## 🧠 How It Works

This app uses the **AdaIN** (Adaptive Instance Normalization) method for fast arbitrary style transfer:

1. A **VGG encoder** extracts features from both the content and style images
2. **AdaIN** aligns the mean and variance of content features to match the style features
3. A **decoder** reconstructs the stylized image from the adapted features

> Based on: *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* — Huang & Belongie (2017)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| Deep Learning | PyTorch, TorchVision |
| Image Processing | Pillow, NumPy |
| Frontend | HTML, Bootstrap |
| Production Server | Gunicorn |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/gajanand1234567/neural-style-transfer-app.git
cd neural-style-transfer-app
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download model weights
Place the following model files in their respective locations:
- `NST_Code/vgg_normalised.pth` — VGG encoder weights
- `NST_Code/experiment/final_exp/decoder_final.pth` — Trained AdaIN decoder

> ⚠️ These files are not included in the repository due to their large size.

### 5. Run the app
```bash
cd NST_Code
python app.py
```

Open your browser at **http://localhost:5000**

---

## 📁 Project Structure

```
neural-style-transfer-app/
├── NST_Code/
│   ├── app.py                  # Flask application
│   ├── train.py                # Model training script
│   ├── vgg_normalised.pth      # VGG encoder weights (not tracked)
│   ├── utils/
│   │   ├── models.py           # VGGEncoder & Decoder architecture
│   │   └── utils.py            # AdaIN helper functions
│   ├── templates/              # HTML templates
│   ├── static/                 # Static assets & uploads
│   ├── examples/               # Example images
│   └── experiment/             # Trained model checkpoints
├── requirements.txt
├── Procfile.txt                # For deployment
└── README.md
```

---

## 📦 Requirements

See [requirements.txt](requirements.txt) for full list. Key dependencies:

- `torch==2.2.2`
- `torchvision==0.17.2`
- `Flask==3.1.2`
- `Pillow==12.0.0`
- `numpy>=1.24,<2.0`

---

## 👤 Author

**Gajanand**  
📧 22ce02008@iitbbs.ac.in  
🔗 [GitHub](https://github.com/gajanand1234567)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).