import random

import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFilter, ImageOps
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Human vs AI Drawing Challenge", page_icon="🎨")

REPO_ID = "zarqankhn/quickdraw-345-tflite"
STROKE_WIDTH = 12

# Simple, easy-to-draw words for "Easy" difficulty
EASY_WORDS = [
    "smiley face", "sailboat", "clock", "cat", "house", "apple", "star",
    "bicycle", "tree", "fish", "airplane", "sun", "umbrella", "moon", "car",
]


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
    ink = ImageOps.invert(gray)

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


def show_guesses(preds, labels, k):
    st.subheader(f"AI's Top {k} Guesses:")
    top = np.argsort(preds)[::-1][:k]
    for idx in top:
        label = labels[idx] if idx < len(labels) else f"Class {idx}"
        conf = float(preds[idx])
        st.write(f"**{label.capitalize()}** — {conf * 100:.1f}%")
        st.progress(min(max(conf, 0.0), 1.0))
    return top


def pick_word(labels, difficulty):
    pool = [w for w in EASY_WORDS if w in labels] if difficulty == "Easy" else labels
    return random.choice(pool or labels)


def new_round(labels, difficulty):
    st.session_state.target = pick_word(labels, difficulty)
    st.session_state.round_id += 1
    st.session_state.scored = False


# ---------------------------------------------------------------- setup
st.title("🎨 Human vs AI: Drawing Challenge")

with st.spinner("Loading Quick, Draw! (345-class) model..."):
    interpreter, input_details, output_details, labels = load_quickdraw_model()
labels_lower = [l.lower() for l in labels]

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["🎯 Challenge", "✏️ Free draw"])
    top_k = st.slider("Number of guesses shown", 3, 10, 3)
    if mode == "🎯 Challenge":
        difficulty = st.radio("Difficulty", ["Easy", "Hard (all 345 words)"],
                              help="Easy = 15 simple words. Hard = anything in the model.")
        diff_key = "Easy" if difficulty == "Easy" else "Hard"
        st.divider()
        st.metric("Score", st.session_state.get("score", 0))
        st.caption(f"Rounds played: {st.session_state.get('rounds', 0)}")
        if st.button("Reset score"):
            st.session_state.score = 0
            st.session_state.rounds = 0
            st.rerun()

# session state defaults
st.session_state.setdefault("score", 0)
st.session_state.setdefault("rounds", 0)
st.session_state.setdefault("round_id", 0)
st.session_state.setdefault("scored", False)
if "target" not in st.session_state:
    st.session_state.target = pick_word(labels_lower, "Easy")

# Re-roll the word if difficulty changed and current word isn't in that pool
if mode == "🎯 Challenge":
    if diff_key == "Easy" and st.session_state.target not in EASY_WORDS:
        new_round(labels_lower, diff_key)

# ---------------------------------------------------------------- header
if mode == "🎯 Challenge":
    st.markdown(f"## Draw: **{st.session_state.target.upper()}**")
    st.write("Can the AI guess it? 3 pts if it's the 1st guess, 2 pts for 2nd, 1 pt for 3rd.")
    canvas_key = f"canvas_{st.session_state.round_id}"
else:
    st.write("Draw any simple doodle below — the AI knows 345 different objects!")
    canvas_key = "canvas_free"
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
    key=canvas_key,
    return_image_data=True,
)

# ---------------------------------------------------------------- buttons
if mode == "🎯 Challenge":
    col1, col2 = st.columns(2)
    predict_clicked = col1.button("Predict Drawing", type="primary", use_container_width=True)
    if col2.button("⏭️ New word / skip", use_container_width=True):
        new_round(labels_lower, diff_key)
        st.rerun()
else:
    predict_clicked = st.button("Predict Drawing", type="primary")

# ---------------------------------------------------------------- predict
if predict_clicked:
    if canvas_result.image_data is None:
        st.warning("Draw something on the canvas first!")
    else:
        processed = preprocess_drawing(canvas_result.image_data)
        if processed is None:
            st.warning("Draw something on the canvas first!")
        else:
            img_array, preview_img = processed
            preds = predict(interpreter, input_details, output_details, img_array)

            if mode == "🎯 Challenge":
                target = st.session_state.target
                ranked = np.argsort(preds)[::-1]
                rank = next((r for r, i in enumerate(ranked[:3]) if labels_lower[i] == target), None)

                if rank is None:
                    st.error(f"😅 The AI didn't get **{target}** in its top 3.")
                    points = 0
                else:
                    points = 3 - rank
                    st.success(f"🎉 The AI guessed **{target}** as guess #{rank + 1}! +{points} pts")
                    if rank == 0:
                        st.balloons()

                if not st.session_state.scored:
                    st.session_state.scored = True
                    st.session_state.rounds += 1
                    st.session_state.score += points
                    st.caption("Score updated — press 'New word / skip' for the next round.")

            show_guesses(preds, labels, top_k)

            with st.expander("See what the AI actually saw (28x28)"):
                st.image(preview_img.resize((112, 112), Image.Resampling.NEAREST), width=112)
