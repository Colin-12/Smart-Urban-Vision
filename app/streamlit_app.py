"""
Smart Urban Vision — Dashboard Streamlit
Personne D : Dashboard & Intégration
========================================
Structure : 6 pages avec fallback sur données fictives.
Pour brancher les vrais fichiers → chercher les blocs marqués
  # ✅ BRANCHER ICI — <description>
et retirer le mock correspondant.
"""

import json
import random
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image

# ─────────────────────────────────────────────
# CHEMINS — toujours relatifs depuis la racine
# ─────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

PATHS = {
    "cnn_model":       ROOT / "models" / "cnn_model.keras",
    "class_names":     ROOT / "models" / "class_names.json",
    "eval_report":     ROOT / "outputs" / "cnn" / "evaluation_report.json",
    "confusion":       ROOT / "outputs" / "cnn" / "confusion_matrix.png",
    "train_curves":    ROOT / "outputs" / "cnn" / "training_curves.png",
    "yolo_model":      ROOT / "models" / "yolo11n.pt",
    "seg_model":       ROOT / "models" / "yolo11n-seg.pt",
    "detections_json": ROOT / "outputs" / "yolo" / "detection_results.json",
    "tracking_stats":  ROOT / "outputs" / "video" / "tracking_stats.json",
    "video_annotated": ROOT / "outputs" / "video" / "annotated_video.mp4",
    "tracking_demo":   ROOT / "outputs" / "video" / "tracking_demo.mp4",
}

# ─────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Urban Vision",
    layout="wide",
    page_icon="🏙️",
    initial_sidebar_state="expanded",
)

# CSS global
st.markdown("""
<style>
    .status-ok   { color: #2ecc71; font-weight: bold; }
    .status-wait { color: #f39c12; font-weight: bold; }
    .mock-banner { background:#2c3e50; color:#f39c12; padding:8px 14px;
                   border-radius:6px; font-size:0.85em; margin-bottom:10px; }
    .metric-card { background:#1e1e2e; border-radius:10px; padding:16px;
                   text-align:center; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS GÉNÉRAUX
# ─────────────────────────────────────────────

def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def file_status(path: Path) -> str:
    return "✅ Disponible" if path.exists() else "⏳ En attente"

def mock_banner(msg="Données fictives — sera remplacé dès que le fichier réel est disponible"):
    st.markdown(f'<div class="mock-banner">🧪 {msg}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 0 — ACCUEIL
# ─────────────────────────────────────────────

def render_home():
    st.title("🏙️ Smart Urban Vision")
    st.subheader("Détection, Segmentation et Analyse Intelligente de Scènes Urbaines")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Ce dashboard regroupe l'ensemble du pipeline de vision par ordinateur développé
        dans le cadre du projet M2 Big Data & IA — Sup de Vinci.

        **Ce qu'il permet :**
        - 🖼️ Classifier une image urbaine (art, nuisance, neutre) via un CNN
        - 🔍 Détecter des objets avec YOLO11
        - ✂️ Segmenter les zones d'intérêt pixel par pixel
        - 🎥 Analyser une vidéo ou un flux en temps réel
        - 📊 Visualiser les performances des modèles
        """)

    with col2:
        st.markdown("### 📦 État des livrables")
        checks = {
            "Modèle CNN":           PATHS["cnn_model"],
            "Modèle YOLO détection":PATHS["yolo_model"],
            "Modèle YOLO seg.":     PATHS["seg_model"],
            "Rapport évaluation":   PATHS["eval_report"],
            "Résultats YOLO":       PATHS["detections_json"],
            "Stats tracking":       PATHS["tracking_stats"],
            "Vidéo annotée":        PATHS["video_annotated"],
        }
        for label, path in checks.items():
            icon = "✅" if path.exists() else "⏳"
            st.write(f"{icon} {label}")

    st.markdown("---")
    st.markdown("**Équipe :**  Personne A — CNN · Personne B — YOLO · Personne C — Vidéo · Personne D — Dashboard")


# ─────────────────────────────────────────────
# PAGE 1 — CLASSIFICATION CNN
# ─────────────────────────────────────────────

@st.cache_resource
def load_cnn_model(path):
    from tensorflow.keras.models import load_model
    return load_model(str(path))

def render_cnn():
    st.header("🖼️ Classification CNN")
    st.markdown("Classifie une image urbaine parmi : **art**, **nuisance**, **neutre**.")

    if PATHS["class_names"].exists():
        class_names = load_json(PATHS["class_names"])
    else:
        class_names = ["art", "nuisance", "neutre"]

    model_ready = PATHS["cnn_model"].exists()

    if not model_ready:
        mock_banner("Modèle CNN pas encore livré (H+8). Interface en mode démo.")

    uploaded = st.file_uploader("📤 Uploader une image urbaine", type=["jpg", "jpeg", "png"])

    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.image(img_pil, caption="Image uploadée", use_column_width=True)

        with col2:
            st.subheader("Résultat de classification")

            if model_ready:
                # ✅ BRANCHER ICI — Modèle CNN réel (livré par Personne A)
                model = load_cnn_model(PATHS["cnn_model"])
                img_resized = img_pil.resize((224, 224))
                arr = np.array(img_resized) / 255.0
                pred = model.predict(arr[np.newaxis, ...])[0]
            else:
                # 🧪 MOCK
                pred = np.random.dirichlet(np.ones(len(class_names)))

            predicted_class = class_names[int(pred.argmax())]
            colors = {"art": "🟢", "nuisance": "🔴", "neutre": "🔵"}

            for cls, prob in zip(class_names, pred):
                st.metric(
                    label=f"{colors.get(cls, '⚪')} {cls.capitalize()}",
                    value=f"{prob * 100:.1f}%"
                )

            st.success(f"**Classe prédite : {predicted_class.upper()}**")

            if not model_ready:
                st.caption("⚠️ Résultat fictif — remplacer par le vrai modèle keras")


# ─────────────────────────────────────────────
# PAGE 2 — DÉTECTION YOLO
# ─────────────────────────────────────────────

def draw_detections_mock(img_pil, detections):
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(img_pil)
    colors_map = {
        "person": "#e74c3c", "car": "#3498db", "bicycle": "#2ecc71",
        "graffiti": "#f39c12", "default": "#9b59b6"
    }
    w, h = img_pil.size
    for det in detections:
        cls = det.get("class_name", "objet")
        conf = det.get("confidence", 0.0)
        bbox = det.get("bbox", [0.1 * w, 0.1 * h, 0.5 * w, 0.5 * h])
        x1, y1, x2, y2 = bbox
        color = colors_map.get(cls, colors_map["default"])
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{cls} {conf:.0%}", color=color,
                fontsize=9, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5, pad=2))
    ax.axis("off")
    return fig


def render_yolo():
    st.header("🔍 Détection YOLO")
    st.markdown("Détecte et localise les objets d'intérêt dans une scène urbaine.")

    yolo_ready = PATHS["yolo_model"].exists()

    if not yolo_ready:
        mock_banner("Modèle YOLO pas encore livré (H+8). Résultats simulés.")

    uploaded = st.file_uploader("📤 Uploader une image", type=["jpg", "jpeg", "png"], key="yolo_upload")
    conf_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.5, 0.05)

    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        w, h = img_pil.size

        if yolo_ready:
            # ✅ BRANCHER ICI — Inférence YOLO réelle (livré par Personne B)
            from ultralytics import YOLO
            model = YOLO(str(PATHS["yolo_model"]))
            results = model(img_pil, conf=conf_threshold)[0]
            detections = []
            for box in results.boxes:
                detections.append({
                    "class_name": results.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()
                })
        else:
            # 🧪 MOCK
            detections = [
                {"class_name": "person",  "confidence": 0.91, "bbox": [w*0.1, h*0.1, w*0.25, h*0.8]},
                {"class_name": "car",     "confidence": 0.85, "bbox": [w*0.4, h*0.3, w*0.75, h*0.7]},
                {"class_name": "graffiti","confidence": 0.73, "bbox": [w*0.6, h*0.05, w*0.95, h*0.5]},
            ]

        detections = [d for d in detections if d["confidence"] >= conf_threshold]

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = draw_detections_mock(img_pil, detections)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col2:
            st.subheader(f"🎯 {len(detections)} objet(s) détecté(s)")
            if detections:
                df = pd.DataFrame(detections)[["class_name", "confidence"]]
                df["confidence"] = df["confidence"].apply(lambda x: f"{x:.0%}")
                df.columns = ["Classe", "Confiance"]
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Aucun objet au-dessus du seuil.")

    st.markdown("---")
    st.subheader("📋 Résultats sur le dataset de test")

    if PATHS["detections_json"].exists():
        # ✅ BRANCHER ICI — detection_results.json livré par Personne B
        results_data = load_json(PATHS["detections_json"])
    else:
        mock_banner("Fichier detection_results.json en attente (H+8).")
        results_data = [
            {"image": "urban_01.jpg", "detections": [
                {"class_name": "person", "confidence": 0.91},
                {"class_name": "car", "confidence": 0.88}
            ], "masks_available": True},
            {"image": "urban_02.jpg", "detections": [
                {"class_name": "bicycle", "confidence": 0.82}
            ], "masks_available": True},
        ]

    all_dets = []
    for item in results_data:
        for det in item.get("detections", []):
            all_dets.append({
                "Image": item["image"],
                "Classe": det.get("class_name", "?"),
                "Confiance": det.get("confidence", 0)
            })

    if all_dets:
        df_all = pd.DataFrame(all_dets)
        df_all["Confiance"] = df_all["Confiance"].apply(lambda x: f"{x:.0%}")

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df_all, use_container_width=True, hide_index=True)
        with col2:
            class_counts = pd.DataFrame(all_dets).groupby("Classe").size().reset_index(name="Nombre")
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.barh(class_counts["Classe"], class_counts["Nombre"], color="#3498db")
            ax2.set_xlabel("Nombre de détections")
            ax2.set_title("Distribution par classe")
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)


# ─────────────────────────────────────────────
# PAGE 3 — SEGMENTATION
# ─────────────────────────────────────────────

def render_segmentation():
    st.header("✂️ Segmentation")
    st.markdown("Passe de la bounding box au **masque pixel par pixel** avec YOLO11-seg.")

    seg_ready = PATHS["seg_model"].exists()

    if not seg_ready:
        mock_banner("Modèle de segmentation pas encore livré (H+14).")

    uploaded = st.file_uploader("📤 Uploader une image", type=["jpg", "jpeg", "png"], key="seg_upload")

    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        img_arr = np.array(img_pil)
        w, h = img_pil.size

        if seg_ready:
            # ✅ BRANCHER ICI — Segmentation réelle (livré par Personne B)
            from ultralytics import YOLO
            model = YOLO(str(PATHS["seg_model"]))
            results = model(img_pil)[0]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(results.plot())
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            # 🧪 MOCK
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(img_arr)
            axes[0].set_title("Image originale")
            axes[0].axis("off")

            overlay = img_arr.copy().astype(float)
            mask_colors = [
                ([int(h*0.1), int(h*0.8), int(w*0.05), int(w*0.3)], [231, 76, 60]),
                ([int(h*0.2), int(h*0.7), int(w*0.35), int(w*0.75)], [52, 152, 219]),
            ]
            for (r1, r2, c1, c2), color in mask_colors:
                overlay[r1:r2, c1:c2] = (
                    overlay[r1:r2, c1:c2] * 0.5 + np.array(color) * 0.5
                )

            axes[1].imshow(overlay.astype(np.uint8))
            axes[1].set_title("Segmentation simulée (mock)")
            axes[1].axis("off")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption("⚠️ Masques fictifs — remplacer par yolo11n-seg.pt quand Personne B livre")

    st.markdown("---")
    st.info("""
    **Pourquoi la segmentation apporte plus qu'une bounding box ?**
    - Permet d'isoler exactement les pixels d'un graffiti sans inclure le mur alentour
    - Utile pour calculer la surface réelle d'une dégradation
    - Nécessaire pour des analyses de couleur ou de texture précises
    """)


# ─────────────────────────────────────────────
# PAGE 4 — VIDÉO & TRACKING
# ─────────────────────────────────────────────

def render_video():
    st.header("🎥 Vidéo & Tracking")
    st.markdown("Analyse d'un flux vidéo urbain avec suivi des objets dans le temps.")

    if PATHS["tracking_stats"].exists():
        # ✅ BRANCHER ICI — tracking_stats.json livré par Personne C
        stats = load_json(PATHS["tracking_stats"])
    else:
        mock_banner("tracking_stats.json en attente (H+12).")
        stats = {
            "unique_objects_tracked": 7,
            "fps_average": 18.5,
            "total_frames": 450,
            "duration_seconds": 24.3,
            "classes_detected": ["person", "car", "bicycle"],
            "class_counts": {"person": 4, "car": 2, "bicycle": 1}
        }

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Objets suivis", stats.get("unique_objects_tracked", "—"))
    col2.metric("⚡ FPS moyen", stats.get("fps_average", "—"))
    col3.metric("🎞️ Frames totales", stats.get("total_frames", "—"))
    col4.metric("⏱️ Durée (s)", stats.get("duration_seconds", "—"))

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📹 Vidéo annotée")
        if PATHS["video_annotated"].exists():
            # ✅ BRANCHER ICI — annotated_video.mp4 livré par Personne C
            st.video(str(PATHS["video_annotated"]))
        else:
            mock_banner("annotated_video.mp4 en attente (H+4)")
            st.image("https://placehold.co/640x360/2c3e50/ecf0f1?text=Video+annotee+a+venir",
                     caption="Placeholder — vidéo avec bounding boxes frame par frame")

    with col_right:
        st.subheader("🔄 Démo Tracking")
        if PATHS["tracking_demo"].exists():
            # ✅ BRANCHER ICI — tracking_demo.mp4 livré par Personne C
            st.video(str(PATHS["tracking_demo"]))
        else:
            mock_banner("tracking_demo.mp4 en attente (H+14)")
            st.image("https://placehold.co/640x360/1a252f/ecf0f1?text=Tracking+Demo+a+venir",
                     caption="Placeholder — démonstration du tracking avec IDs d'objets")

    st.markdown("---")
    st.subheader("📊 Objets détectés dans la vidéo")

    class_counts = stats.get("class_counts", {})
    if class_counts:
        df_classes = pd.DataFrame(
            list(class_counts.items()), columns=["Classe", "Occurrences"]
        ).sort_values("Occurrences", ascending=False)

        col_a, col_b = st.columns(2)
        with col_a:
            st.dataframe(df_classes, use_container_width=True, hide_index=True)
        with col_b:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(df_classes["Classe"], df_classes["Occurrences"],
                   color=["#e74c3c", "#3498db", "#2ecc71"][:len(df_classes)])
            ax.set_ylabel("Occurrences")
            ax.set_title("Objets détectés")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ─────────────────────────────────────────────
# PAGE 5 — PERFORMANCES
# ─────────────────────────────────────────────

def render_stats():
    st.header("📊 Performances des modèles")
    st.subheader("🖼️ Classification CNN")

    if PATHS["eval_report"].exists():
        # ✅ BRANCHER ICI — evaluation_report.json livré par Personne A
        report = load_json(PATHS["eval_report"])
    else:
        mock_banner("evaluation_report.json en attente (H+12).")
        report = {
            "accuracy": 0.87,
            "classes": ["art", "nuisance", "neutre"],
            "per_class": {
                "art":     {"precision": 0.90, "recall": 0.85, "f1": 0.87},
                "nuisance":{"precision": 0.83, "recall": 0.88, "f1": 0.85},
                "neutre":  {"precision": 0.86, "recall": 0.82, "f1": 0.84},
            }
        }

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Accuracy globale", f"{report['accuracy'] * 100:.1f}%")
    col2.metric("Nombre de classes", len(report.get("classes", [])))
    avg_f1 = np.mean([v["f1"] for v in report["per_class"].values()])
    col3.metric("F1 moyen", f"{avg_f1:.2f}")

    df_perf = pd.DataFrame(report["per_class"]).T
    df_perf = df_perf.applymap(lambda x: f"{x:.2f}")
    st.dataframe(df_perf.style.highlight_max(axis=0), use_container_width=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Matrice de confusion")
        if PATHS["confusion"].exists():
            # ✅ BRANCHER ICI — confusion_matrix.png livré par Personne A
            st.image(str(PATHS["confusion"]), use_column_width=True)
        else:
            mock_banner("confusion_matrix.png en attente (H+12).")
            fig, ax = plt.subplots(figsize=(4, 4))
            classes = ["art", "nuisance", "neutre"]
            cm = np.array([[34, 3, 2], [4, 38, 2], [1, 2, 37]])
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
            ax.set_xlabel("Prédit")
            ax.set_ylabel("Réel")
            ax.set_title("Matrice de confusion (mock)")
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > 20 else "black")
            fig.colorbar(im)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with col_b:
        st.subheader("Courbes d'entraînement")
        if PATHS["train_curves"].exists():
            # ✅ BRANCHER ICI — training_curves.png livré par Personne A
            st.image(str(PATHS["train_curves"]), use_column_width=True)
        else:
            mock_banner("training_curves.png en attente (H+12).")
            epochs = list(range(1, 21))
            train_acc  = [min(0.45 + 0.025*e - 0.0005*e**2, 0.95) for e in epochs]
            val_acc    = [min(0.42 + 0.022*e - 0.0006*e**2, 0.88) for e in epochs]
            train_loss = [max(1.0  - 0.04*e  + 0.001*e**2,  0.12) for e in epochs]
            val_loss   = [max(1.05 - 0.038*e + 0.0012*e**2, 0.18) for e in epochs]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
            ax1.plot(epochs, train_acc,  label="Train", color="#2ecc71")
            ax1.plot(epochs, val_acc,    label="Val",   color="#3498db", linestyle="--")
            ax1.set_title("Accuracy")
            ax1.set_xlabel("Époque")
            ax1.legend()
            ax2.plot(epochs, train_loss, label="Train", color="#e74c3c")
            ax2.plot(epochs, val_loss,   label="Val",   color="#f39c12", linestyle="--")
            ax2.set_title("Loss")
            ax2.set_xlabel("Époque")
            ax2.legend()
            fig.suptitle("Courbes d'entraînement (mock)", fontsize=10)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ─────────────────────────────────────────────
# SIDEBAR & ROUTING
# ─────────────────────────────────────────────

def main():
    st.sidebar.title("🏙️ Smart Urban Vision")
    st.sidebar.markdown("---")

    pages = {
        "🏠 Accueil":            render_home,
        "🖼️ Classification CNN": render_cnn,
        "🔍 Détection YOLO":     render_yolo,
        "✂️ Segmentation":       render_segmentation,
        "🎥 Vidéo & Tracking":   render_video,
        "📊 Performances":       render_stats,
    }

    page = st.sidebar.radio("Navigation", list(pages.keys()))

    st.sidebar.markdown("---")
    st.sidebar.markdown("**État des fichiers**")
    quick_checks = {
        "CNN Model":   PATHS["cnn_model"],
        "YOLO Model":  PATHS["yolo_model"],
        "Eval Report": PATHS["eval_report"],
        "Tracking":    PATHS["tracking_stats"],
    }
    for label, path in quick_checks.items():
        icon = "🟢" if path.exists() else "🟡"
        st.sidebar.caption(f"{icon} {label}")

    pages[page]()


if __name__ == "__main__":
    main()