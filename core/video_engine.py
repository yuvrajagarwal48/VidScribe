"""
Video Processing Engine
-----------------------
Handles video file inspection, scene detection via PySceneDetect,
middle keyframe extraction to JPEG, and frame-accurate timestamp mathematics.
"""

import os
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scenedetect import detect, ContentDetector

import config
from schemas.video_models import SceneBoundary, KeyframeData


class VideoEngine:
    """
    Core video processing class for scene segmentation and keyframe extraction.
    """

    def __init__(
        self,
        video_path: str,
        threshold: float = config.SCENE_THRESHOLD,
        min_scene_len: int = config.MIN_SCENE_LEN
    ):
        """
        Initialize the VideoEngine with a video source path.
        
        Args:
            video_path: Absolute or relative file path to the video file
            threshold: Sensitivity threshold for PySceneDetect ContentDetector
            min_scene_len: Minimum scene duration in frames
        """
        self.video_path = str(video_path)
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found at: {self.video_path}")

        # Unique identifier derived from filename (e.g. 'cricket_highlights')
        self.video_id = Path(self.video_path).stem
        self.threshold = threshold
        self.min_scene_len = min_scene_len

        # Directory dedicated to storing extracted keyframe images for this video
        self.keyframes_dir = config.KEYFRAMES_DIR / self.video_id
        self.keyframes_dir.mkdir(parents=True, exist_ok=True)

        # Extract baseline video properties using OpenCV
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or self.fps is None:
            self.fps = 30.0  # Safe default to avoid division by zero
        
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_seconds = self.frame_count / self.fps if self.fps > 0 else 0.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Cached scene boundaries and keyframes
        self._scenes: List[SceneBoundary] = []
        self._keyframes: Dict[int, KeyframeData] = {}

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """
        Converts seconds into standard formatted time string: HH:MM:SS.mmm
        """
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        msec = int((seconds % 1) * 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d}.{msec:03d}"

    def detect_scenes(self) -> List[SceneBoundary]:
        """
        Executes PySceneDetect ContentDetector to find scene boundaries.
        Returns a list of structured SceneBoundary objects.
        """
        if self._scenes:
            return self._scenes

        try:
            raw_scene_list = detect(
                self.video_path,
                ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_len)
            )
        except Exception as e:
            print(f"Warning: Scene detection error ({e}). Falling back to single scene.")
            raw_scene_list = []

        # If no scenes were detected (e.g. static video or continuous shot), treat whole video as 1 scene
        if not raw_scene_list:
            start_frame = 0
            end_frame = self.frame_count if self.frame_count > 0 else int(self.fps * 10)
            boundary = SceneBoundary(
                scene_index=0,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=0.0,
                end_time=self.duration_seconds,
                formatted_start=self.format_timestamp(0.0),
                formatted_end=self.format_timestamp(self.duration_seconds),
                duration=self.duration_seconds
            )
            self._scenes = [boundary]
            return self._scenes

        # Transform PySceneDetect output into typed SceneBoundary models
        self._scenes = []
        for idx, (start_tc, end_tc) in enumerate(raw_scene_list):
            start_frame = getattr(start_tc, "frame_num", getattr(start_tc, "get_frames", lambda: 0)())
            end_frame = getattr(end_tc, "frame_num", getattr(end_tc, "get_frames", lambda: 0)())
            start_sec = getattr(start_tc, "seconds", getattr(start_tc, "get_seconds", lambda: 0.0)())
            end_sec = getattr(end_tc, "seconds", getattr(end_tc, "get_seconds", lambda: 0.0)())
            duration = end_sec - start_sec

            self._scenes.append(
                SceneBoundary(
                    scene_index=idx,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=start_sec,
                    end_time=end_sec,
                    formatted_start=self.format_timestamp(start_sec),
                    formatted_end=self.format_timestamp(end_sec),
                    duration=duration
                )
            )

        return self._scenes

    def extract_keyframes(self) -> Dict[int, KeyframeData]:
        """
        Extracts the middle frame for each detected scene and saves it as a JPEG.
        Returns a mapping from scene_index to KeyframeData.
        """
        if self._keyframes:
            return self._keyframes

        scenes = self.detect_scenes()
        cap = cv2.VideoCapture(self.video_path)

        for scene in scenes:
            # Target the middle frame of the scene for the most representative visual sample
            middle_frame = scene.start_frame + (scene.end_frame - scene.start_frame) // 2
            middle_sec = middle_frame / self.fps

            keyframe_filename = f"{self.video_id}_scene_{scene.scene_index:03d}.jpg"
            keyframe_path = str(self.keyframes_dir / keyframe_filename)

            # Seek directly to frame and save image if not already generated
            if not os.path.exists(keyframe_path):
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                success, frame = cap.read()
                if success and frame is not None:
                    is_ok, buffer = cv2.imencode(".jpg", frame)
                    if is_ok:
                        with open(keyframe_path, "wb") as f_out:
                            f_out.write(buffer)
                else:
                    print(f"Warning: Failed to extract frame {middle_frame} for scene {scene.scene_index}")

            self._keyframes[scene.scene_index] = KeyframeData(
                scene_index=scene.scene_index,
                image_path=keyframe_path,
                middle_frame=middle_frame,
                middle_time=middle_sec,
                formatted_middle=self.format_timestamp(middle_sec)
            )

        cap.release()
        return self._keyframes

    def extract_single_keyframe(self, timestamp_seconds: float) -> Optional[str]:
        """
        On-demand frame extractor: Grabs a single frame at an exact timestamp.
        Used for dynamic zooming / frame inspection without processing the entire video.
        """
        cap = cv2.VideoCapture(self.video_path)
        frame_idx = int(timestamp_seconds * self.fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        cap.release()

        if not success or frame is None:
            return None

        out_path = str(self.keyframes_dir / f"ondemand_{int(timestamp_seconds * 1000)}.jpg")
        is_ok, buffer = cv2.imencode(".jpg", frame)
        if is_ok:
            with open(out_path, "wb") as f_out:
                f_out.write(buffer)
            return out_path
        return None
