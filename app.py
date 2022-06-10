#!/usr/bin/env python

from __future__ import annotations

import argparse

import gradio as gr

from model import Model

DESCRIPTION = '''# MangaLineExtraction_PyTorch

This is an unofficial demo for [https://github.com/ljsabc/MangaLineExtraction_PyTorch](https://github.com/ljsabc/MangaLineExtraction_PyTorch).
'''
FOOTER = '<img id="visitor-badge" alt="visitor badge" src="https://visitor-badge.glitch.me/badge?page_id=hysts.mangalineextraction_pytorch" />'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    return parser.parse_args()


def main():
    args = parse_args()
    model = Model(device=args.device)

    with gr.Blocks(theme=args.theme, css='style.css') as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    input_image = gr.Image(label='Input', type='numpy')
                    run_button = gr.Button(value='Run')
            with gr.Column():
                result = gr.Image(label='Result',
                                  type='numpy',
                                  elem_id='result')

        gr.Markdown(FOOTER)

        run_button.click(fn=model.predict, inputs=input_image, outputs=result)

    demo.launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
