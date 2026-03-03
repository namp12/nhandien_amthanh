import torch
from pyannote.audio import Pipeline, Model, Inference
from scipy.spatial.distance import cdist
import os
import requests
from pydub import AudioSegment
import numpy as np

class MeetingAudioProcessor:
    def __init__(self, hf_token: str = None, viettel_key: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.viettel_key = viettel_key
        
        # 1. Model Diarization (Phân tách người nói)
        self.diarize_pipeline = None
        if hf_token:
            self.diarize_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
            )
            if self.device == "cuda":
                self.diarize_pipeline.to(torch.device("cuda"))

        # 2. Model Embedding (Để nhận diện giọng nói cụ thể của ai)
        self.embedding_model = Model.from_pretrained(
            "pyannote/embedding", use_auth_token=hf_token
        )
        self.inference = Inference(self.embedding_model, window="whole")

    def stt_viettel(self, segment_audio_path: str):
        """Gửi đoạn âm thanh nhỏ lên Viettel AI"""
        url = "https://viettelai.vn/asr/v1/speech-to-text"
        headers = {'token': self.viettel_key}
        with open(segment_audio_path, 'rb') as f:
            files = {'file': f}
            try:
                response = requests.post(url, headers=headers, files=files)
                if response.status_code == 200:
                    return response.json().get("results", [{}])[0].get("text", "")
            except:
                pass
        return ""

    def identify_speaker(self, segment_embedding, reference_embeddings):
        """So sánh vân tay giọng nói với danh sách mẫu (Nam, Thầy, ...)"""
        # reference_embeddings là Dict: {"Nam": embedding_vector, "Thầy": embedding_vector}
        min_dist = 100
        best_name = "Unknown"
        
        for name, ref_emb in reference_embeddings.items():
            dist = cdist([segment_embedding], [ref_emb], metric="cosine")[0][0]
            if dist < 0.5 and dist < min_dist: # Ngưỡng 0.5 là tiêu chuẩn, có thể chỉnh lại
                min_dist = dist
                best_name = name
        return best_name

    def process_audio(self, audio_path: str, references=None):
        # references: {"Nam": "path/to/nam_voice.wav", "Thầy": "path/to/thay_voice.wav"}
        
        # Bước 1: Lấy vân tay của những người mẫu (Nếu có)
        ref_embs = {}
        if references:
            for name, path in references.items():
                ref_embs[name] = self.inference(path)

        # Bước 2: Phân tách người nói trong cuộc họp
        diarization = self.diarize_pipeline(audio_path)
        audio = AudioSegment.from_file(audio_path)
        
        final_transcript = []
        # Gom các đoạn nói của cùng 1 người lại để xử lý hoặc xử lý từng đoạn
        for turn, _, speaker_id in diarization.itertracks(yield_label=True):
            # Cắt đoạn âm thanh
            start_ms = turn.start * 1000
            end_ms = turn.end * 1000
            chunk = audio[start_ms:end_ms]
            
            temp_chunk_path = f"temp_{speaker_id}_{int(start_ms)}.wav"
            chunk.export(temp_chunk_path, format="wav")
            
            # Bước 3: Định danh (Nói là Speaker_01 nhưng thực chất là AI?)
            seg_emb = self.inference(temp_chunk_path)
            real_name = self.identify_speaker(seg_emb, ref_embs)
            display_name = real_name if real_name != "Unknown" else speaker_id
            
            # Bước 4: Chuyển văn bản bằng Viettel STT
            text = self.stt_viettel(temp_chunk_path)
            
            if text.strip():
                final_transcript.append({
                    "speaker": display_name,
                    "text": text,
                    "start": turn.start,
                    "end": turn.end
                })
            
            # Xóa file tạm
            if os.path.exists(temp_chunk_path):
                os.remove(temp_chunk_path)
                
        return final_transcript

# Ví dụ sử dụng:
# processor = MeetingAudioProcessor(hf_token="YOUR_HF_TOKEN")
# transcript = processor.process_audio("meeting.mp3")
# print(transcript)
