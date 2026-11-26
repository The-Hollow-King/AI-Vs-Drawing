import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFilter, ImageOps
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Human vs AI Drawing Challenge", page_icon="🎨")

REPO_ID = "zarqankhn/quickdraw-345-tflite"
STROKE_WIDTH = 12


@st.cache_resource
def load_quickdraw_model():
    model_path = hf_hub_download(repo_id=REPO_ID, filename="quickdraw_model.tflite")
    labels_path = hf_hub_download(repo_id=REPO_ID, filename="labels.txt")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip().replace("_", " ") for line in f if line.strip()]

    return interpreter, input_details, output_details, labels


def preprocess_drawing(image_data):
    """Canvas -> 28x28 bitmap with thickened white strokes on black."""
    rgba = Image.fromarray(image_data.astype("uint8"), "RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    gray = Image.alpha_composite(white, rgba).convert("L")
    ink = ImageOps.invert(gray)  # strokes bright, background 0

    mask = ink.point(lambda p: 255 if p > 20 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return None

    cropped = ink.crop(bbox)
    w, h = cropped.size
    side = max(w, h)
    pad = max(4, int(side * 0.08))
    size = side + 2 * pad

    square = Image.new("L", (size, size), 0)
    square.paste(cropped, ((size - w) // 2, (size - h) // 2))

    mid = square.resize((112, 112), Image.Resampling.LANCZOS)
    mid = mid.filter(ImageFilter.MaxFilter(5))  # thicken strokes
    small = mid.resize((28, 28), Image.Resampling.BOX)
    small = ImageOps.autocontrast(small)

    arr = np.array(small, dtype="float32") / 255.0
    return arr, small


def predict(interpreter, input_details, output_details, arr):
    detail = input_details[0]
    x = arr.reshape(tuple(detail["shape"]))
    if detail["dtype"] == np.uint8:
        x = (x * 255.0).astype(np.uint8)
    else:
        x = x.astype(detail["dtype"])

    interpreter.set_tensor(detail["index"], x)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0].astype("float32")

    if preds.max() > 1.0 or preds.min() < 0.0 or abs(preds.sum() - 1.0) > 1e-2:
        e = np.exp(preds - preds.max())
        preds = e / e.sum()
    return preds


st.title("🎨 Human vs AI: Drawing Challenge")
st.write("Draw any simple doodle below — the AI knows 345 different objects!")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of guesses", 1, 10, 3)

with st.spinner("Loading Quick, Draw! (345-class) model..."):
    interpreter, input_details, output_details, labels = load_quickdraw_model()

with st.expander("See all 345 things the AI can recognize"):
    st.write(", ".join(sorted(labels)))

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=STROKE_WIDTH,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=520,
    width=520,
    drawing_mode="freedraw",
    key="canvas",
    return_image_data=True,
)

if st.button("Predict Drawing", type="primary"):
    if canvas_result.image_data is None:
        st.warning("Draw something on the canvas first!")
    else:
        processed = preprocess_drawing(canvas_result.image_data)
        if processed is None:
            st.warning("Draw something on the canvas first!")
        else:
            img_array, preview_img = processed
            preds = predict(interpreter, input_details, output_details, img_array)

            st.subheader(f"AI's Top {top_k} Guesses:")
            for idx in np.argsort(preds)[::-1][:top_k]:
                label = labels[idx] if idx < len(labels) else f"Class {idx}"
                confidence = float(preds[idx])
                st.write(f"**{label.capitalize()}** — {confidence * 100:.1f}%")
                st.progress(min(max(confidence, 0.0), 1.0))

            with st.expander("See what the AI actually saw (28x28)"):
                st.image(preview_img.resize((112, 112), Image.Resampling.NEAREST), width=112)
