"""
Audio Processing & Speech-to-Text Engine
----------------------------------------
Extracts audio from video segments and runs OpenAI Whisper speech recognition
to generate timestamp-synchronized spoken dialogue transcripts.
"""

import os
import tempfile
from typing import List, Optional
from pathlib import Path

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except Exception:
    HAS_FASTER_WHISPER = False

import config
from schemas.video_models import TranscriptSegment


class AudioEngine:
    """
    Manages audio extraction and speech transcription using faster-whisper (CTranslate2 INT8)
    or OpenAI Whisper fallback.
    """

    _whisper_instance = None  # Singleton model instance to prevent repeated weight loads
    _is_faster_whisper = False

    def __init__(self, model_size: str = config.WHISPER_MODEL):
        """
        Initializes the AudioEngine without blocking on Whisper model weight download.
        Whisper will be loaded lazily on the first transcription call.
        """
        self.model_size = model_size

    @classmethod
    def _ensure_whisper_loaded(cls):
        """Loads Whisper model once across all instances (prefers faster-whisper INT8 for speed & low RAM)."""
        if cls._whisper_instance is None:
            if HAS_FASTER_WHISPER:
                print(f"Loading faster-whisper CTranslate2 model ({config.WHISPER_MODEL}, int8, cpu_threads=1)...")
                try:
                    cls._whisper_instance = WhisperModel(
                        config.WHISPER_MODEL,
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=1
                    )
                    cls._is_faster_whisper = True
                    print("faster-whisper CTranslate2 model loaded successfully.")
                    return
                except Exception as e:
                    print(f"Notice: faster-whisper failed to load ({e}). Falling back to standard whisper.")

            import whisper
            print(f"Loading standard Whisper model ({config.WHISPER_MODEL})...")
            cls._whisper_instance = whisper.load_model(config.WHISPER_MODEL)
            cls._is_faster_whisper = False
            print("Standard Whisper model loaded successfully.")

    @staticmethod
    def _extract_audio_subclip(video_path: str, start_time: float, end_time: float, output_wav: str) -> bool:
        """
        Slices audio from video between start_time and end_time using MoviePy.
        Compatible with both MoviePy 1.x and 2.x.
        """
        try:
            try:
                from moviepy import VideoFileClip
            except ImportError:
                try:
                    from moviepy.video.io.VideoFileClip import VideoFileClip
                except ImportError:
                    from moviepy.editor import VideoFileClip

            clip = VideoFileClip(video_path)
            if clip.audio is None:
                clip.close()
                return False

            # Clip slice
            sub = clip.subclipped(start_time, end_time) if hasattr(clip, "subclipped") else clip.subclip(start_time, end_time)
            sub.audio.write_audiofile(
                output_wav,
                codec='pcm_s16le',
                logger=None,
                fps=16000  # 16kHz sample rate optimal for Whisper
            )
            sub.close()
            clip.close()
            return True
        except Exception as e:
            print(f"Warning: Audio extraction failed from {start_time}s to {end_time}s: {e}")
            return False

    def transcribe_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float
    ) -> List[TranscriptSegment]:
        """
        Transcribes a specific temporal segment of the video.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            List of TranscriptSegment objects
        """
        # Duration check: Skip near-zero segments
        if (end_time - start_time) < 0.2:
            return []

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            tmp_wav_path = tmp_audio.name

        try:
            has_audio = self._extract_audio_subclip(video_path, start_time, end_time, tmp_wav_path)
            if not has_audio or not os.path.exists(tmp_wav_path) or os.path.getsize(tmp_wav_path) == 0:
                return []

            # Execute Whisper transcription
            self._ensure_whisper_loaded()
            transcripts: List[TranscriptSegment] = []

            if self._is_faster_whisper:
                segments_gen, _ = self._whisper_instance.transcribe(tmp_wav_path, beam_size=1, temperature=0.0)
                for seg in segments_gen:
                    text = seg.text.strip()
                    if text:
                        seg_start = start_time + float(seg.start)
                        seg_end = start_time + min(float(seg.end), end_time - start_time)
                        transcripts.append(
                            TranscriptSegment(
                                transcript=text,
                                start_time=seg_start,
                                end_time=seg_end,
                                confidence=1.0
                            )
                        )
            else:
                result = self._whisper_instance.transcribe(tmp_wav_path, fp16=False)
                raw_segments = result.get("segments", [])
                if raw_segments:
                    for seg in raw_segments:
                        text = seg.get("text", "").strip()
                        if text:
                            seg_start = start_time + seg.get("start", 0.0)
                            seg_end = start_time + seg.get("end", end_time - start_time)
                            transcripts.append(
                                TranscriptSegment(
                                    transcript=text,
                                    start_time=seg_start,
                                    end_time=seg_end,
                                    confidence=1.0
                                )
                            )
                elif result.get("text", "").strip():
                    transcripts.append(
                        TranscriptSegment(
                            transcript=result["text"].strip(),
                            start_time=start_time,
                            end_time=end_time,
                            confidence=1.0
                        )
                    )

            return transcripts

        finally:
            if os.path.exists(tmp_wav_path):
                try:
                    os.remove(tmp_wav_path)
                except Exception:
                    pass

    def transcribe_full_video(self, video_path: str) -> List[TranscriptSegment]:
        """
        Rapid Tier 1 pass: Transcribes the entire video audio track in a single pass.
        Returns timestamped dialogue segments across the whole video in seconds.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            tmp_wav_path = tmp_audio.name

        try:
            try:
                from moviepy import VideoFileClip
            except ImportError:
                try:
                    from moviepy.video.io.VideoFileClip import VideoFileClip
                except ImportError:
                    from moviepy.editor import VideoFileClip

            clip = VideoFileClip(video_path)
            if clip.audio is None:
                clip.close()
                return []

            clip.audio.write_audiofile(tmp_wav_path, codec='pcm_s16le', logger=None, fps=16000)
            clip.close()

            self._ensure_whisper_loaded()
            transcripts: List[TranscriptSegment] = []

            if self._is_faster_whisper:
                segments_gen, _ = self._whisper_instance.transcribe(tmp_wav_path, beam_size=1, temperature=0.0)
                for seg in segments_gen:
                    text = seg.text.strip()
                    if text:
                        transcripts.append(
                            TranscriptSegment(
                                transcript=text,
                                start_time=float(seg.start),
                                end_time=float(seg.end),
                                confidence=1.0
                            )
                        )
            else:
                result = self._whisper_instance.transcribe(tmp_wav_path, fp16=False)
                for seg in result.get("segments", []):
                    text = seg.get("text", "").strip()
                    if text:
                        transcripts.append(
                            TranscriptSegment(
                                transcript=text,
                                start_time=float(seg.get("start", 0.0)),
                                end_time=float(seg.get("end", 0.0)),
                                confidence=1.0
                            )
                        )

            return transcripts

        finally:
            if os.path.exists(tmp_wav_path):
                try:
                    os.remove(tmp_wav_path)
                except Exception:
                    pass
