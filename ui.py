import os
import argparse
import random
import gradio as gr
from main import load_models, cache_path
from PIL import Image
from os import path
from vlm import LLMUtility

canvas_size = 512
llm_obj = LLMUtility()

def score_fn(image):
    global llm_obj
    response, rating =  llm_obj.get_response_score(image)
    return response

if not path.exists(cache_path):
    os.makedirs(cache_path, exist_ok=True)

with gr.Blocks() as demo:
    infer = load_models()

    with gr.Column():
        with gr.Row():
            with gr.Column():
                s = gr.Slider(label="steps", minimum=4, maximum=8,
                              step=1, value=4, interactive=True)
                c = gr.Slider(label="cfg", minimum=0.1, maximum=3,
                              step=0.1, value=1, interactive=True)
                i_s = gr.Slider(label="sketch strength", minimum=0.1,
                                maximum=0.9, step=0.1, value=0.9, interactive=True)
            with gr.Column():
                mod = gr.Text(label="Model Hugging Face id (after changing this wait until the model downloads in the console)",
                              value="Lykon/dreamshaper-7", interactive=True)
                t = gr.Text(
                    label="Prompt", value="8K, realistic, colorful art, natural", interactive=True)
            with gr.Column():
                score = gr.Text(
                                label="Judge", value="0/10", interactive=True)
        with gr.Row(equal_height=True):
            i = gr.Image(source="canvas", tool="color-sketch", shape=(canvas_size,
                         canvas_size), width=canvas_size, height=canvas_size, type="pil")
            o = gr.Image(width=canvas_size, height=canvas_size)
            
            def process_image(p, im, steps, cfg, image_strength, seed):
                if not im:
                    return Image.new("RGB", (canvas_size, canvas_size))

                gen_img = infer(
                    prompt=p,
                    image=im,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    strength=image_strength,
                    seed=1337
                )
                score = score_fn(image=gen_img)
                return gen_img, score

            reactive_controls = [t, i, s, c, i_s]

            for control in reactive_controls:
                control.change(fn=process_image,
                               inputs=reactive_controls, outputs=[o, score])

            def update_model(model_name):
                global infer
                infer = load_models(model_name)

            mod.change(fn=update_model, inputs=mod)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # If the option python ui.py --share is attached, it will be deployed to Gradio
    parser.add_argument("--share", action="store_true",
                        help="Deploy on Gradio for sharing", default=False)
    args = parser.parse_args()
    demo.launch(share=args.share)

