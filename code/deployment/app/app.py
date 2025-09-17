import io
import os
import requests
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Animals10 Classifier", page_icon="🐾", layout="centered")

API_URL = os.getenv("API_URL", "http://api:8000")


@st.cache_data(show_spinner=False)
def get_labels():
    try:
        resp = requests.get(f"{API_URL}/labels", timeout=5)
        resp.raise_for_status()
        return resp.json().get("labels", [])
    except Exception:
        return []


st.title("Animals10 Classifier 🐾")
st.caption("Upload an image of an animal and get the predicted class.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", width='stretch')
    except Exception:
        st.error("Could not read the image.")
        image = None

    if image is not None and st.button("Predict"):
        with st.spinner("Calling API..."):
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            buf.seek(0)
            files = {"file": (uploaded.name, buf, "image/jpeg")}
            try:
                resp = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                st.error(f"API error: {e}")
                data = None

        if data:
            st.success(f"Prediction: {data.get('predicted_label')}")
            top3 = data.get("top3", [])
            if top3:
                st.subheader("Top 3")
                for item in top3:
                    st.write(f"- {item['label']}: {item['score']:.3f}")

st.divider()
labels = get_labels()
if labels:
    with st.expander("Class labels"):
        st.write(", ".join(labels))
else:
    st.info("API labels not available yet. Make sure API is running.")


