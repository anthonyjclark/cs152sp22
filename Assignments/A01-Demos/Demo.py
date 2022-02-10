"""
ssh -L PORT:localhost:PORT dgx01
streamlit run Demo.py --server.port PORT
interpreter: /opt/mambaforge/envs/cs152/bin/python
"""

from base64 import b64decode
from inspect import getmodule
from io import BytesIO
from typing import Hashable
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import streamlit as st
from torch import hub
from transformers import pipeline
from urllib.request import urlopen


class HashableModel:
    def __init__(self, name, loader) -> None:
        self.name = name
        self.model = loader(name)

    def __call__(self, *args, **kwds):
        return self.model(*args, **kwds)


def load_animegan2(name):
    """Return a model that takes an input and return an image."""
    generator = hub.load(
        "AK391/animegan2-pytorch:main",
        "generator",
        pretrained=True,
        device="cuda",
        progress=True,
    )

    face2paint = hub.load(
        "AK391/animegan2-pytorch:main",
        "face2paint",
        size=512,
        device="cuda",
        side_by_side=False,
    )

    def model(img):
        return face2paint(generator, img)

    return model


@st.cache(hash_funcs={HashableModel: lambda m: m.name})
def load_model(name):
    return model_loaders[name]()


model_loaders = {
    "AK391/animegan2": lambda: HashableModel("animegan2", load_animegan2),
    "fill-mask": lambda: HashableModel("fill-mask", pipeline),
    "image-classification": lambda: HashableModel("image-classification", pipeline),
    "image-segmentation": lambda: HashableModel("image-segmentation", pipeline),
    "object-detection": lambda: HashableModel("object-detection", pipeline),
    "sentiment-analysis": lambda: HashableModel("sentiment-analysis", pipeline),
    "text-generation": lambda: HashableModel("text-generation", pipeline),
    "translation_en_to_fr": lambda: HashableModel("translation_en_to_fr", pipeline),
}

st.title("Neural Network Demos")
option = st.selectbox(
    "Choose a demo type from this list.", sorted(list(model_loaders.keys())), index=5
)

model = load_model(option)

if option == "sentiment-analysis":
    input_text = st.text_input("Input text for sentiment analysis.")
    if input_text:
        output_classification = model(input_text)[0]
        label = output_classification["label"]
        score = output_classification["score"]
        output_message = f"The sentiment is **{label}** with a score of `{score}`."
        output_message

elif option == "fill-mask":
    mask_token = model.model.tokenizer.mask_token
    input_text = st.text_input(
        f"Write a sentence that includes '{mask_token}' (without the quotes) and this model will fill in the blank."
    )
    if input_text:
        output_messages = model(input_text)

        output_message = "\n\n".join(
            input_text.replace(mask_token, f"**{out['token_str'].strip()}**")
            for out in output_messages
        )
        output_message

elif option == "text-generation":
    input_text = st.text_input("Start a sentence and let the model finish it...")
    if input_text:
        output_sentence = model(input_text)[0]
        output_sentence["generated_text"]

elif option == "translation_en_to_fr":
    input_text = st.text_input("This model translates English to French.")
    if input_text:
        output_translation = model(input_text)[0]
        output_translation["translation_text"]

elif option == "image-classification":
    input_url = st.text_input("Provide the URI for an image (jpg or png).")
    if input_url:
        image_classifications = model(input_url)

        st.image(input_url, caption="Original Image.")

        "### Top five predictions for the image"

        predictions = [
            f"1. Predicting **{out['label']}** with a score of `{out['score']:.2f}`."
            for out in image_classifications
        ]
        output_message = "\n".join(predictions)
        output_message

elif option == "image-segmentation":
    input_url = st.text_input("Provide the URI for an image (jpg or png).")
    if input_url:
        image_segmentation = model(input_url)[0]
        segmented_image = Image.open(BytesIO(b64decode(image_segmentation["mask"])))
        col1, col2 = st.columns(2)
        col1.image(input_url, caption="Original Image.")
        col2.image(segmented_image, caption="Segmented Image.")

elif option == "object-detection":
    input_url = st.text_input("Provide the URI for an image (jpg or png).")
    # "https://storage.googleapis.com/petbacker/images/blog/2017/dog-and-cat-cover.jpg"
    if input_url:

        detected_objects = model(input_url)

        img = Image.open(urlopen(input_url))

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(img)

        for out in detected_objects:
            x, y = out["box"]["xmin"], out["box"]["ymin"]
            width = out["box"]["xmax"] - out["box"]["xmin"]
            height = out["box"]["ymax"] - out["box"]["ymin"]
            ax.add_patch(
                Rectangle(
                    (x, y), width, height, linewidth=1, edgecolor="r", facecolor="none"
                )
            )
            ax.text(x, y + 20, f"{out['label']} ({out['score']:.2f})")

        ax.set_axis_off()

        st.pyplot(fig)

elif option == "AK391/animegan2":
    input_url = st.text_input("Provide the URI for an image (jpg or png).")
    if input_url:
        img = Image.open(urlopen(input_url))
        styled_image = model(img)

        col1, col2 = st.columns(2)
        col1.image(input_url, caption="Original Image.")
        col2.image(styled_image, caption="Styled Image.")

        "Style model from: https://huggingface.co/spaces/akhaliq/AnimeGANv2"
