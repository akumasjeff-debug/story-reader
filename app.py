import os
import uuid
import asyncio
import io
import hashlib
import re
from collections import Counter

from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image
from deep_translator import GoogleTranslator
import edge_tts

if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)


def _load_env():
    path = os.path.join(BASE_DIR, 'config.env')
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

# ── OCR ENGINE ──────────────────────────────────────────────
GEMINI_KEY      = os.environ.get('GEMINI_API_KEY', '')
# 依序嘗試，前面的失敗（如模型下架 404）就自動降級到下一個
GEMINI_MODELS   = ['gemini-2.5-flash', 'gemini-3.5-flash',
                   'gemini-2.0-flash', 'gemini-flash-latest']
_gemini_ok      = None   # 記住上次成功的模型名稱，下次優先用
_easyocr_reader = None


def _gemini_generate(parts):
    """依序嘗試可用模型，回傳第一個成功的 response。"""
    global _gemini_ok
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_KEY)
    # 上次成功的模型排最前面
    candidates = ([_gemini_ok] if _gemini_ok else []) + \
                 [m for m in GEMINI_MODELS if m != _gemini_ok]
    last_err = None
    for name in candidates:
        try:
            resp = genai.GenerativeModel(name).generate_content(parts)
            if name != _gemini_ok:
                print(f"Gemini 使用模型：{name}")
            _gemini_ok = name
            return resp
        except Exception as e:
            last_err = e
            print(f"模型 {name} 失敗，嘗試下一個：{e}")
    raise last_err if last_err else RuntimeError('無可用的 Gemini 模型')


def _easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        print("載入 EasyOCR 模型（首次需下載 ~1.5GB，請稍候）...")
        import easyocr
        _easyocr_reader = easyocr.Reader(['ch_tra', 'en'], gpu=False)
        print("EasyOCR 載入完成！")
    return _easyocr_reader


def _sort(results):
    if not results:
        return []
    s = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))
    lines, cur, cy = [], [s[0]], s[0][0][0][1]
    for item in s[1:]:
        iy = item[0][0][1]
        if abs(iy - cy) < 30:
            cur.append(item)
        else:
            lines.append(sorted(cur, key=lambda r: r[0][0][0]))
            cur, cy = [item], iy
    if cur:
        lines.append(sorted(cur, key=lambda r: r[0][0][0]))
    return [x for line in lines for x in line]


def _ocr(img: Image.Image) -> str:
    if GEMINI_KEY:
        resp = _gemini_generate([
            "Extract all text from this image exactly as it appears. "
            "Return only the raw text, preserve paragraph breaks, "
            "no explanations or extra content.",
            img
        ])
        return resp.text.strip()
    else:
        import numpy as np
        results = _easyocr().readtext(np.array(img), paragraph=False)
        return ' '.join(r[1] for r in _sort(results) if r[2] > 0.3)


# ── SCAN CACHE (FEAT-7) ─────────────────────────────────────
_scan_cache = {}


# ── TEMP CLEANUP (BUG-3) ────────────────────────────────────
def _cleanup_temp():
    import time
    now = time.time()
    cutoff = 24 * 3600
    for fname in os.listdir(TEMP_DIR):
        fpath = os.path.join(TEMP_DIR, fname)
        try:
            if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > cutoff:
                os.remove(fpath)
        except Exception:
            pass


# ── FLASK APP ────────────────────────────────────────────────
app = Flask(__name__)

VOICES = {
    'zh-TW-female': 'zh-TW-HsiaoChenNeural',
    'zh-TW-male':   'zh-TW-YunJheNeural',
    'en-US-female': 'en-US-JennyNeural',
    'en-US-male':   'en-US-GuyNeural',
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def status():
    return jsonify({'ocr': 'gemini' if GEMINI_KEY else 'easyocr'})


@app.route('/api/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return jsonify({'error': '無圖片'}), 400
    try:
        img = Image.open(io.BytesIO(request.files['image'].read())).convert('RGB')
        # Resize if oversized
        if max(img.width, img.height) > 1920:
            r = 1920 / max(img.width, img.height)
            img = img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)
        return jsonify({'text': _ocr(img)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/translate', methods=['POST'])
def translate():
    text = (request.json or {}).get('text', '').strip()
    if not text:
        return jsonify({'translated': ''})
    try:
        return jsonify({'translated': GoogleTranslator(source='auto', target='zh-TW').translate(text)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/synthesize', methods=['POST'])
def synthesize():
    data      = request.json or {}
    text      = data.get('text', '').strip()
    voice_key = data.get('voice', 'zh-TW-female')
    rate      = data.get('rate', '-25%')
    volume    = data.get('volume', '+50%')

    if not text:
        return jsonify({'error': '無文字'}), 400

    voice    = VOICES.get(voice_key, VOICES['zh-TW-female'])
    audio_id = str(uuid.uuid4())
    path     = os.path.join(TEMP_DIR, f'{audio_id}.mp3')
    timings  = []
    chunks   = []

    async def run():
        communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
        async for chunk in communicate.stream():
            if chunk['type'] == 'audio':
                chunks.append(chunk['data'])
            elif chunk['type'] == 'WordBoundary':
                timings.append({
                    'text':     chunk['text'],
                    'offset':   chunk['offset']   / 10_000_000,
                    'duration': chunk['duration'] / 10_000_000,
                })

    # BUG-4: 使用獨立 event loop 避免衝突
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run())
    finally:
        loop.close()

    with open(path, 'wb') as f:
        f.write(b''.join(chunks))

    return jsonify({'audio_id': audio_id, 'timings': timings})


@app.route('/api/audio/<audio_id>')
def get_audio(audio_id):
    safe = ''.join(c for c in audio_id if c.isalnum() or c == '-')
    path = os.path.join(TEMP_DIR, f'{safe}.mp3')
    if not os.path.exists(path):
        return jsonify({'error': 'Not found'}), 404
    return send_file(path, mimetype='audio/mpeg')


# FEAT-8：長句自動再切
def _sub_split(text):
    if len(text) <= 80:
        return [text]
    for sep in [', ', ' and ', ' but ', '，', '，但', '，而']:
        idx = text.find(sep, 30)  # 至少保留 30 字
        if 30 < idx < len(text) - 10:
            left = text[:idx + (1 if sep == '，' else len(sep))].strip()
            right = text[idx + (1 if sep == '，' else len(sep)):].strip()
            if left and right:
                return [left] + _sub_split(right)
    return [text]


def _split_segments(en_text: str, zh_text: str) -> list:
    def split(text):
        parts = re.split(r'(?<=[.!?。！？])\s+', text.strip())
        if len(parts) <= 1 and '\n' in text:
            parts = text.split('\n')
        result = []
        for p in parts:
            p = p.strip()
            if p:
                result.extend(_sub_split(p))
        return result

    en_parts = split(en_text) if en_text else []
    zh_parts = split(zh_text) if zh_text else []
    count = max(len(en_parts), len(zh_parts)) if (en_parts or zh_parts) else 0

    return [
        {'en': en_parts[i] if i < len(en_parts) else '',
         'zh': zh_parts[i] if i < len(zh_parts) else ''}
        for i in range(count)
        if (en_parts[i] if i < len(en_parts) else '') or (zh_parts[i] if i < len(zh_parts) else '')
    ]


# FEAT-6：常用字彙整（停用詞集合）
_STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'it', 'to', 'of',
    'in', 'and', 'or', 'but', 'on', 'at', 'for', 'i', 'you', 'he',
    'she', 'we', 'they', 'this', 'that', 'with', 'as', 'be', 'have',
    'had', 'do', 'did', 'not', 'my', 'your', 'his', 'her', 'our', 'their'
}


def _extract_vocab(original: str) -> list:
    words = re.findall(r'[a-zA-Z]+', original)
    filtered = [w.lower() for w in words if w.lower() not in _STOPWORDS]
    top5 = [word for word, _ in Counter(filtered).most_common(5)]
    result = []
    for word in top5:
        try:
            zh = GoogleTranslator(source='en', target='zh-TW').translate(word)
        except Exception:
            zh = ''
        result.append({'en': word, 'zh': zh or ''})
    return result


@app.route('/api/scan', methods=['POST'])
def scan():
    if 'image' not in request.files:
        return jsonify({'error': '無圖片'}), 400
    try:
        image_bytes = request.files['image'].read()

        # FEAT-7：md5 快取，相同圖片直接回傳
        md5 = hashlib.md5(image_bytes).hexdigest()
        if md5 in _scan_cache:
            return jsonify(_scan_cache[md5])

        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if max(img.width, img.height) > 1920:
            r = 1920 / max(img.width, img.height)
            img = img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)

        original   = _ocr(img)
        translated = GoogleTranslator(source='auto', target='zh-TW').translate(original) if original.strip() else ''
        segments   = _split_segments(original, translated)
        vocab      = _extract_vocab(original) if original.strip() else []

        payload = {
            'original':   original,
            'translated': translated,
            'segments':   segments,
            'vocab':      vocab,
        }
        _scan_cache[md5] = payload
        return jsonify(payload)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


_cleanup_temp()

IS_CLOUD = bool(os.environ.get('RENDER') or os.environ.get('RAILWAY_ENVIRONMENT'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))

    if IS_CLOUD:
        print(f"Cloud mode, port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        import socket
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            local_ip = '127.0.0.1'
        engine = 'Gemini Flash' if GEMINI_KEY else 'EasyOCR'
        print(f"OCR 引擎：{engine}")
        print(f"本機：  https://127.0.0.1:{port}")
        print(f"手機：  https://{local_ip}:{port}  (同一 WiFi)")
        app.run(host='0.0.0.0', port=port, debug=False, ssl_context='adhoc')
