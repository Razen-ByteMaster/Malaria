# 🦟 Malaria Cell-Image Classifier

> Upload a blood-smear cell photo — get back **Malaria** or **No Malaria** with a confidence score.
> A fine-tuned ResNet50 behind a tiny FastAPI service. Small code, serious model. 🔬

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green?logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow)
![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)

## ✨ What is this?

A deep-learning image classifier that detects malaria parasites in thin blood-smear cell photos. You send an image to one endpoint — it sends back a diagnosis-style prediction with confidence.

| Capability | Details |
|---|---|
| 🧠 Model | ResNet50, fine-tuned for binary classification (`Malaria` / `No Malaria`) |
| 🖼️ Input | Any common image format (JPEG/PNG) — auto-resized to 224×224, ImageNet-normalized |
| 📊 Output | Predicted class + softmax confidence as a percentage |
| ⚡ Device | CUDA when available, CPU otherwise — no code change needed |
| 🚀 API | Single `POST /predict` endpoint, JSON in / JSON out |

> 📈 **Test accuracy: XX.X%** — replace with your measured number before publishing.

## ⚙️ How it works

```
Upload (JPEG/PNG)
      ↓
Resize 224×224 → ToTensor → Normalize (ImageNet mean/std)
      ↓
ResNet50 (custom 2-class head, malaria.pth weights)
      ↓
Softmax → argmax → { "prediction": "Malaria", "confidence": 97.42 }
```

## 🚀 Quickstart

```bash
cd Malaria
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn app:app --reload
```

> ⚠️ The trained weights file `malaria.pth` (~94 MB) must sit next to `app.py`. See [Model weights](#-model-weights) before pushing to GitHub.

Test it (server runs at http://127.0.0.1:8000 by default):

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@cell_image.png"
```

Interactive docs for free at **http://127.0.0.1:8000/docs** 🎉

## 🔍 Sample output

```json
{
  "filename": "cell_image.png",
  "prediction": "Malaria",
  "confidence": 97.42
}
```

Low confidence? Treat it like a lab assistant raising its hand — confirm with a second test, don't blindly trust the number. ⚠️

## 🛠️ Tech stack

| Layer | Tech | Why |
|---|---|---|
| Model | ResNet50 (torchvision) | Strong transfer-learning baseline for medical images |
| Framework | PyTorch | Training + inference in one ecosystem |
| API | FastAPI | Async uploads, auto OpenAPI docs at `/docs` |
| Preprocessing | torchvision transforms + Pillow | Resize, tensor, ImageNet normalize |
| Validation | FastAPI `UploadFile` | Streaming multipart uploads |

## 🧬 Dataset

Trained on the NIH malaria blood-smear cell-image dataset (mirrored on Kaggle) — segmented red-blood-cell photos labeled parasitized vs. uninfected.

- Kaggle mirror: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
- Original source: National Library of Medicine (NIH)

## 🏋️ Model weights

- `malaria.pth` is a full ResNet50 state dict (~94 MB) — that's just under GitHub's 100 MB per-file limit.
- Recommended: keep it **out of git** (uncomment the `*.pth` line in `.gitignore`) and attach it to a **GitHub Release** instead; link the release in this section.
- At runtime the file must be named exactly `malaria.pth` and sit next to `app.py`.

## 📁 Project structure

```
Malaria/
├── app.py               # FastAPI app: model load + POST /predict
├── malaria.pth          # Fine-tuned ResNet50 weights (~94 MB, see above)
├── requirements.txt     # Python dependencies
└── README.md            # You are here 👋
```

## 🗺️ Roadmap

- [ ] `GET /health` endpoint (model loaded? device?)
- [ ] Input validation (reject non-images with 400 instead of 500)
- [ ] Confidence-threshold flag (`review_warning` when unsure)
- [ ] Batch prediction endpoint
- [ ] Dockerfile + pinned `torch` CUDA/CPU builds
- [ ] Grad-CAM heatmaps (show *where* the parasite is)

## 📄 License

MIT. Built by [Razen-ByteMaster](https://github.com/Razen-ByteMaster) as a portfolio project. PRs welcome! 🎉

> ⚕️ **Disclaimer:** educational/portfolio project — not a medical device. Never use it for real diagnosis.
