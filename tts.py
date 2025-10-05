import os
import argparse
import time
import io
import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer
from flask import Flask, request, jsonify, send_file

# ----------------------------
# Flask-приложение
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Загрузка модели
# ----------------------------
def load_model(model_name="facebook/mms-tts-rus", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Загрузка модели: {model_name}")
    print(f"[INFO] Устройство: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name).to(device)
    model.eval()

    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[INFO] Используется GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")

    return tokenizer, model, device

# ----------------------------
# Синтез речи
# ----------------------------
@torch.inference_mode()
def synthesize(text: str, tokenizer, model, device):
    if not text.strip():
        raise ValueError("Пустой текст.")

    start_time = time.time()
    inputs = tokenizer(text, return_tensors="pt").to(device)
    out = model(**inputs)
    waveform = out.waveform.cpu().squeeze(0)
    sr = getattr(model.config, "sampling_rate", 16000)
    duration = time.time() - start_time

    print(f"[INFO] Синтез завершён за {duration:.2f} с (длина: {len(waveform)/sr:.2f} с)")

    return waveform, sr

# ----------------------------
# HTTP-endpoint
# ----------------------------
import os
import time
import io
import soundfile as sf
from flask import Flask, request, jsonify, send_file

@app.route("/tts", methods=["POST"])
def tts_route():
    data = request.get_json()
    if data is None or "text" not in data:
        return jsonify({"error": "Не передан текст"}), 400

    text = data["text"]
    try:
        waveform, sr = synthesize(text, app.tokenizer, app.model, app.device)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # --- подготовка пути для сохранения ---
    os.makedirs("/app/output", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"/app/output/tts_{timestamp}.wav"

    # --- сохраняем на диск ---
    sf.write(out_path, waveform.numpy(), sr)
    print(f"[OK] Сохранено в {out_path}")

    # --- готовим буфер для ответа ---
    buf = io.BytesIO()
    sf.write(buf, waveform.numpy(), sr, format="WAV")
    buf.seek(0)

    return send_file(
        buf,
        mimetype="audio/wav",
        as_attachment=False,
        download_name="tts.wav",
    )

# ----------------------------
# CLI-режим
# ----------------------------
def cli_mode(text, out_path, tokenizer, model, device):
    waveform, sr = synthesize(text, tokenizer, model, device)
    sf.write(out_path, waveform.numpy(), sr)
    print(f"[OK] Сохранено в {out_path}")

# ----------------------------
# Точка входа
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Запустить HTTP-сервер")
    parser.add_argument("--text", type=str, help="Текст для синтеза (CLI-режим)")
    parser.add_argument("--out", type=str, help="Файл для сохранения WAV (CLI-режим)")
    parser.add_argument("--model", type=str, default="facebook/mms-tts-rus", help="Имя модели")
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    app.tokenizer, app.model, app.device = tokenizer, model, device

    if args.serve:
        print("[INFO] HTTP-сервер запущен на http://0.0.0.0:5000")
        app.run(host="0.0.0.0", port=5000)
    else:
        if not args.text or not args.out:
            parser.error("--text и --out обязательны в CLI-режиме")
        cli_mode(args.text, args.out, tokenizer, model, device)

if __name__ == "__main__":
    main()
