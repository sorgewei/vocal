import gradio as gr
import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram
import torchaudio


model = SimpleAudioModel()
model.load_state_dict(torch.load('audio_model.pth'))
model.eval()

# 语音合成函数
def text_to_speech(text):
   
    waveform = torch.randn(1, 16000)  
    return waveform.numpy()


iface = gr.Interface(
    fn=text_to_speech,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Enter text here..."),
    outputs="audio",
    title="Text to Speech",
    description="Enter text and get audio output."
)


iface.launch()
