import re
import torch
import soundfile as sf
from .text_normalizer import normalize_text
from .codec import CodecModel, load_audio
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gender_id = { "female":"<|FEMALE|>" ,"male":"<|MALE|>" }

class VachaSpeech:
    def __init__(
        self,
        model_id="VIZINTZOR/VachaSpeech"
    ):
        self.codec_model = CodecModel.from_pretrained(model_id).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map=device
        ).eval().to(device)
        
    def generate(self, text, gender="female", temperature=0.8, top_p=0.95, top_k=40, repetition_penalty=1.1):
        clean_text = normalize_text(text)
        predict_max_len = len(clean_text) * 5

        messages = [
            {"role": "user", "content": gender_id[gender] + " " + clean_text}
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(input_text, 
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=512).to(device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=predict_max_len,
                min_new_tokens=10,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        result = self.tokenizer.decode(generated, skip_special_tokens=True)
        codes = list(map(int, re.findall(r"<\|s_(\d+)\|>", result)))
        return codes
    
    def decode(self, codes, ref_audio, output="output.wav"):
        if not isinstance(codes, torch.Tensor):
            codes = torch.tensor(codes, device=device)
        else:
            codes = codes.to(device)
        waveform = load_audio(ref_audio, sample_rate=self.codec_model.config.sample_rate).to(device)
        with torch.no_grad():
            features = self.codec_model.encode(waveform, return_content=False)
            resynth = self.codec_model.decode(
                content_token_indices=codes,
                global_embedding=features.global_embedding,
            )
        sf.write(output, resynth.cpu().numpy(), self.codec_model.config.sample_rate)
        return output