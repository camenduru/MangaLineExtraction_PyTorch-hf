#!/usr/bin/env python

from __future__ import annotations

import gradio as gr

from model import Model

DESCRIPTION = '''# MangaLineExtraction_PyTorch

This is an unofficial demo for [https://github.com/ljsabc/MangaLineExtraction_PyTorch](https://github.com/ljsabc/MangaLineExtraction_PyTorch).
'''

model = Model()

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label='Input', type='numpy')
            run_button = gr.Button(value='Run')
        with gr.Column():
            result = gr.Image(label='Result', type='numpy', elem_id='result')

    run_button.click(fn=model.predict, inputs=input_image, outputs=result)

demo.queue().launch(show_api=False)
