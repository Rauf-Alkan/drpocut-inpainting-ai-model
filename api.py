from flask import Flask, request, send_file, jsonify
import subprocess
import os
import uuid

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/inpaint', methods=['POST'])
def inpaint():
    if 'video' not in request.files or 'mask' not in request.files:
        return jsonify({'error': 'video ve mask gerekli'}), 400
    job_id = str(uuid.uuid4())
    os.makedirs(f'/tmp/{job_id}', exist_ok=True)
    video_path = f'/tmp/{job_id}/video.mp4'
    mask_path = f'/tmp/{job_id}/mask.mp4'
    request.files['video'].save(video_path)
    request.files['mask'].save(mask_path)
    result = subprocess.run(['python3', 'run_diffueraser.py', '--input_video', video_path, '--input_mask', mask_path, '--save_path', f'/tmp/{job_id}'], capture_output=True, text=True)
    output_path = f'/tmp/{job_id}/diffueraser_result.mp4'
    if not os.path.exists(output_path):
        return jsonify({'error': result.stderr}), 500
    return send_file(output_path, mimetype='video/mp4')

app.run(host='0.0.0.0', port=8080)