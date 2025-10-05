import os
import argparse
import torch
from transformers import VitsModel, AutoTokenizer
import soundfile as sf
from flask import Flask, request, jsonify, send_file
import io

app = Flask(__name__)

def load_model(model_name="facebook/mms-tts-rus", device=None, use_fp16=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


@torch.inference_mode()
def synthesize(text, tokenizer, model, device, use_fp16=True):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.cuda.amp.autocast(enabled=use_fp16 and device.type == "cuda"):
        out = model(**inputs)
    waveform = out.waveform.cpu().squeeze(0)
    sr = getattr(model.config, "sampling_rate", 16000)
    return waveform, sr


@app.route("/tts", methods=["POST"])
def tts_route():
    data = request.get_json()
    if data is None or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    waveform, sr = synthesize(
        text, app.tokenizer, app.model, app.device, use_fp16=True
    )

    buf = io.BytesIO()
    sf.write(buf, waveform.numpy(), sr, format="WAV")
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav", as_attachment=False, download_name="tts.wav")

def cli_mode(text, out_path, tokenizer, model, device):
    waveform, sr = synthesize(text, tokenizer, model, device)
    sf.write(out_path, waveform.numpy(), sr)
    print(f"Saved to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Run HTTP server")
    parser.add_argument("--text", type=str, help="Text to synthesize (CLI mode)")
    parser.add_argument("--out", type=str, help="Output wav path")
    parser.add_argument("--model", type=str, default="facebook/mms-tts-rus")
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16")
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model, use_fp16=not args.no_fp16)
    app.tokenizer = tokenizer
    app.model = model
    app.device = device

    if args.serve:
        app.run(host="0.0.0.0", port=5000)
    else:
        if args.text is None or args.out is None:
            parser.error("In non-serve mode, --text and --out must be provided")
        cli_mode(args.text, args.out, tokenizer, model, device)

if __name__ == "__main__":
    main()
