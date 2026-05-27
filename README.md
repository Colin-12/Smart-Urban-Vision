# 🏙️ Smart Urban Vision

> Détection, Segmentation et Analyse Intelligente de Scènes Urbaines  
> Projet M2 Big Data & IA — Sup de Vinci

---

## 🎯 Description

Pipeline complet de vision par ordinateur appliqué à des scènes urbaines :
classification CNN, détection YOLO, segmentation et analyse vidéo,
le tout accessible via un dashboard Streamlit.

## 🚀 Installation

```bash
git clone https://github.com/[votre-repo]/smart-urban-vision
cd smart-urban-vision
pip install -r app/requirements.txt
```

> **Modèles lourds** (`.keras`, `.pt`, vidéos) : télécharger depuis le Drive partagé
> 📁 [Lien Google Drive — à compléter par Personne A]
> et placer les fichiers dans `models/` et `outputs/video/` selon la structure ci-dessous.

## ▶️ Lancement

```bash
streamlit run app/streamlit_app.py
```

## 📁 Structure du projet

```
smart-urban-vision/
├── app/
│   ├── streamlit_app.py         ← Dashboard principal
│   ├── requirements.txt
│   └── utils/
├── models/                      ← Modèles (Drive)
│   ├── cnn_model.keras
│   ├── class_names.json
│   ├── yolo11n.pt
│   └── yolo11n-seg.pt
├── outputs/
│   ├── cnn/                     ← Métriques CNN
│   ├── yolo/                    ← Résultats détection/segmentation
│   └── video/                   ← Vidéos annotées et stats
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_cnn.ipynb
│   ├── 03_yolo_detection.ipynb
│   ├── 04_segmentation.ipynb
│   ├── 05_video.ipynb
│   └── 06_tracking.ipynb
└── presentation/
    └── smart_urban_vision.pdf
```

## 👥 Équipe

| Personne | Rôle |
|----------|------|
| Personne A | Preprocessing & Classification CNN |
| Personne B | Détection YOLO & Segmentation |
| Personne C | Vidéo & Tracking |
| Personne D | Dashboard & Intégration |
