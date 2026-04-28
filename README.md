# VachaSpeech
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/VIZINTZOR/VachaSpeech)

VachaSpeech เป็นระบบ Text-to-Speech (TTS) ภาษาไทย ที่รองรับ Voice Cloning โดยใช้โมเดล LLM เพื่อสร้างเสียงพูดที่เป็นธรรมชาติ ขนาด 0.6B

## โมเดล

| Model Name | Parameters | License | Codec |
|---|---|---|---|
| [VachaSpeech-0.6B](https://huggingface.co/VIZINTZOR/VachaSpeech) | 0.6B | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) | [MioCodec-25Hz-44.1kHz-v2](https://huggingface.co/Aratako/MioCodec-25Hz-44.1kHz-v2) 

## การติดตั้ง 

```bash
pip install git+https://github.com/VYNCX/VachaSpeech.git
```

## ใช้งาน

```python
from vachaspeech import VachaSpeech

tts = VachaSpeech()

text = "วันนี้อากาศปลอดโปร่ง ลมพัดเย็นสบาย รู้สึกเหมาะกับการออกไปเดินเล่นหรือจิบกาแฟข้างนอกมากเลย"
output = tts.generate(text, gender="female")
# โคลนเสียง
tts.decode(output, ref_audio="sample_1.wav", output="output.wav")
```
