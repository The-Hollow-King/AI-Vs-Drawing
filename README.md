# AI-Vs-Drawing
An app I made for college level science exhibition. Will probably add more functionality to this in the future. It is a browser-based drawing game where you sketch a doodle and an AI tries to guess what it is. The app is built with Streamlit and uses a TensorFlow Lite model trained on Google's [Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset, which covers **345 object categories**.

<!-- Add a screenshot or GIF here: ![Demo](demo.png) -->

## Features

- **Challenge mode**: the app gives you a random word to draw. Points are awarded by the AI's guess rank (3 pts for 1st, 2 pts for 2nd, 1 pt for 3rd), with a running score.
- **Free draw mode**: draw anything and see the AI's top guesses (up to 10) with confidence bars.
- **Difficulty levels**: *Easy* (15 simple words) or *Hard* (all 345 categories).
- **"What the AI saw" view**: shows the 28×28 image that was actually fed to the model.

## Tech Stack

| Area | Technology |
|---|---|
| Language | Python |
| Web app / UI | [Streamlit](https://streamlit.io/) |
| Drawing canvas | [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas) |
| Model inference | TensorFlow Lite (`tf.lite.Interpreter`) |
| Model hosting | [Hugging Face Hub](https://huggingface.co/zarqankhn/quickdraw-345-tflite) (`huggingface_hub`) |
| Image processing | Pillow (PIL) |
| Numerical computing | NumPy |
| Dataset | Google Quick, Draw! |

## How It Works

1. The user draws on a canvas.
2. The drawing is preprocessed to match the Quick, Draw! training format:
   - composited on white, then inverted to white strokes on a black background
   - cropped to the drawing's bounding box, centered, and padded into a square
   - strokes thickened so they survive downscaling
   - resized to **28×28** grayscale and normalized to `[0, 1]`
3. The image is passed to the TFLite model, and softmax is applied if the output is not already probabilities.
4. The top-k predictions are displayed. In Challenge mode they are checked against the target word to score points.

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python -m streamlit run app.py
```

The model and labels are downloaded automatically from Hugging Face the first time you run the app.

> **Note:** `streamlit-drawable-canvas` can break on newer Streamlit versions (an `image_to_url` error). If that happens, install `streamlit==1.38.0` or use the `streamlit-drawable-canvas-fix` fork.

## Tips for Best Results

- Draw large and simple, filling most of the canvas.
- Draw only the object, with no background or text.
- Quick, Draw! sketches are fast, simple doodles, so don't over-detail.

## Acknowledgements

- [Google Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset)
- Pretrained model: [`zarqankhn/quickdraw-345-tflite`](https://huggingface.co/zarqankhn/quickdraw-345-tflite) on Hugging Face

## License

Add your license here (for example, MIT).

