"""
Media Synthesis Engine
----------------------
Handles media production workflows:
1. Google Text-to-Speech (gTTS) audio narration synthesis.
2. MoviePy summary video compilation (slicing key moments, stitching montage, overlaying voiceover).
3. Matplotlib/ReportLab multi-page Storyboard PDF generation.
"""

import os
import cv2
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
from gtts import gTTS
import textwrap

import config


class MediaSynthesisEngine:
    """
    Synthesizes creative video summaries, voiceover narrations, and PDF storyboards.
    """

    def __init__(self, output_dir: str = str(config.SUMMARIES_DIR)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_narration(self, text: str, video_id: str) -> Optional[str]:
        """
        Synthesizes speech audio from text using gTTS.
        
        Args:
            text: Narration text to be spoken
            video_id: Video identifier
            
        Returns:
            Path to the saved MP3 narration file
        """
        if not text or not text.strip():
            return None

        # Clean text of markdown artifacts for smooth speech synthesis
        clean_text = text.replace("*", "").replace("#", "").replace("-", " ")
        clean_text = clean_text[:2000]  # Cap length for concise narration

        out_path = str(self.output_dir / f"{video_id}_narration.mp3")
        try:
            tts = gTTS(text=clean_text, lang='en', slow=False)
            tts.save(out_path)
            return out_path
        except Exception as e:
            print(f"Warning: gTTS narration synthesis failed: {e}")
            return None

    def compile_summary_video(
        self,
        video_id: str,
        video_path: str,
        scene_segments: List[Dict[str, Any]],
        narration_text: Optional[str] = None
    ) -> Optional[str]:
        """
        Assembles a short highlights video from the original video using MoviePy,
        overlaying the generated TTS voiceover.
        
        Args:
            video_id: Video identifier
            video_path: Path to the original full video
            scene_segments: List of scenes with 'start_time' and 'end_time'
            narration_text: Optional text to speak over the highlights
            
        Returns:
            Path to generated summary MP4
        """
        if not os.path.exists(video_path) or not scene_segments:
            print("Notice: Missing video file or segments for summary montage.")
            return None

        output_mp4 = str(self.output_dir / f"{video_id}_summary.mp4")

        try:
            # Handle MoviePy 2.x and 1.x imports
            try:
                from moviepy import VideoFileClip, concatenate_videoclips, AudioFileClip
            except ImportError:
                try:
                    from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
                except ImportError:
                    from moviepy.video.io.VideoFileClip import VideoFileClip
                    from moviepy.video.compositing.concatenate import concatenate_videoclips
                    from moviepy.audio.io.AudioFileClip import AudioFileClip

            clips = []
            source_clip = VideoFileClip(video_path)
            total_dur = source_clip.duration

            # Take top 3 to 5 scenes
            selected = scene_segments[:5]
            for seg in selected:
                st = seg.get("start_time", 0.0)
                et = seg.get("end_time", st + 5.0)
                
                # Bounds check
                st = max(0.0, min(st, total_dur - 1.0))
                et = max(st + 1.0, min(et, total_dur))
                
                sub = source_clip.subclipped(st, et) if hasattr(source_clip, "subclipped") else source_clip.subclip(st, et)
                clips.append(sub)

            if not clips:
                source_clip.close()
                return None

            # Concatenate scene clips
            final_video = concatenate_videoclips(clips, method="compose")

            # Attach voiceover narration if available
            narration_path = None
            if narration_text:
                narration_path = self.generate_narration(narration_text, video_id)

            if narration_path and os.path.exists(narration_path):
                try:
                    narration_audio = AudioFileClip(narration_path)
                    # Loop or trim audio to match video duration
                    if narration_audio.duration > final_video.duration:
                        narration_audio = narration_audio.subclipped(0, final_video.duration) if hasattr(narration_audio, "subclipped") else narration_audio.subclip(0, final_video.duration)
                    
                    final_video = final_video.with_audio(narration_audio) if hasattr(final_video, "with_audio") else final_video.set_audio(narration_audio)
                except Exception as e:
                    print(f"Notice: Audio attachment note: {e}")

            # Render output
            final_video.write_videofile(
                output_mp4,
                codec="libx264",
                audio_codec="aac",
                logger=None,
                fps=24
            )

            # Cleanup
            for c in clips:
                c.close()
            final_video.close()
            source_clip.close()

            return output_mp4

        except Exception as e:
            print(f"Error compiling summary video: {e}")
            return None

    def create_storyboard_pdf(
        self,
        video_id: str,
        summary_text: str,
        scenes: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Creates an exportable, publication-quality Storyboard PDF document with
        scene keyframes, timestamps, and commentary using Matplotlib PdfPages.
        
        Args:
            video_id: Video identifier
            summary_text: Executive summary of the video
            scenes: List of scene dictionaries with 'keyframe_path', 'formatted_start', and text
            output_path: Optional custom path for output PDF
            
        Returns:
            Path to the saved PDF file
        """
        pdf_path = output_path or str(self.output_dir / f"{video_id}_storyboard.pdf")

        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages

            with PdfPages(pdf_path) as pdf:
                # ----------------------------------------------------
                # Page 1: Title & Executive Summary Cover
                # ----------------------------------------------------
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')

                # Header Title
                ax.text(0.5, 0.90, "VidScribe Video Storyboard", fontsize=24, fontweight='bold', ha='center', color='#1E293B')
                ax.text(0.5, 0.85, f"Video ID: {video_id}", fontsize=14, ha='center', color='#64748B')
                ax.plot([0.1, 0.9], [0.82, 0.82], color='#38BDF8', lw=2)

                # Summary Section
                ax.text(0.1, 0.77, "Executive Summary", fontsize=16, fontweight='bold', color='#0F172A')
                wrapped_summary = textwrap.fill(summary_text, width=80)
                ax.text(0.1, 0.73, wrapped_summary, fontsize=11, va='top', linespacing=1.6, color='#334155')

                # Scene index stats
                ax.text(0.1, 0.20, f"Total Key Scenes Extracted: {len(scenes)}", fontsize=12, fontweight='bold', color='#475569')
                ax.text(0.1, 0.15, "Generated automatically by VidScribe Agentic VideoRAG Engine.", fontsize=10, style='italic', color='#94A3B8')

                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                # ----------------------------------------------------
                # Subsequent Pages: 2 Scenes Per Page with Keyframe Images
                # ----------------------------------------------------
                scenes_per_page = 2
                for page_start in range(0, len(scenes), scenes_per_page):
                    batch = scenes[page_start:page_start + scenes_per_page]
                    fig, axes = plt.subplots(len(batch), 2, figsize=(8.5, 11), gridspec_kw={'width_ratios': [1.2, 1]})

                    if len(batch) == 1:
                        axes = [axes]  # Normalize 1D array to 2D for consistent indexing

                    for row_idx, scene in enumerate(batch):
                        img_ax = axes[row_idx][0]
                        txt_ax = axes[row_idx][1]

                        img_ax.axis('off')
                        txt_ax.axis('off')

                        # Draw Keyframe Image
                        kf_path = scene.get("keyframe_path") or scene.get("source")
                        if kf_path and os.path.exists(kf_path):
                            try:
                                img = cv2.imread(kf_path)
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img_ax.imshow(img_rgb)
                            except Exception:
                                img_ax.text(0.5, 0.5, "Image preview unavailable", ha='center', va='center')
                        else:
                            img_ax.text(0.5, 0.5, "Keyframe not found", ha='center', va='center')

                        # Scene Details & Narrative
                        scene_no = scene.get("scene_index", page_start + row_idx + 1)
                        t_range = f"{scene.get('formatted_start', '00:00')} - {scene.get('formatted_end', '00:00')}"
                        
                        txt_ax.text(0.05, 0.90, f"Scene {scene_no}", fontsize=14, fontweight='bold', color='#0F172A')
                        txt_ax.text(0.05, 0.80, f"Timestamp: {t_range}", fontsize=11, fontweight='semibold', color='#0284C7')

                        desc = scene.get("document", "") or scene.get("description", "No description available.")
                        wrapped_desc = textwrap.fill(desc, width=40)
                        txt_ax.text(0.05, 0.70, wrapped_desc, fontsize=9.5, va='top', color='#334155', linespacing=1.4)

                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

            return pdf_path

        except Exception as e:
            print(f"Error creating storyboard PDF: {e}")
            return None
