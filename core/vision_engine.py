"""
Vision & OCR Engine
-------------------
Extracts visual features from keyframe images:
1. Optical Character Recognition (OCR) via Tesseract for on-screen text, slides, and scoreboards.
2. Multimodal Object Detection and Spatial Relationship extraction via the pluggable LLMFactory.
"""

import os
import cv2
import json
import base64
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image
import pytesseract
from pydantic import BaseModel, Field

import config
from schemas.video_models import OCRToken, DetectedObject, ObjectSpatialRelation
from core.llm_factory import LLMFactory, extract_text_from_message_content


class VisionDetectionSchema(BaseModel):
    """Pydantic model for strict structured output from multimodal vision LLMs."""
    objects: List[DetectedObject] = Field(
        default_factory=list,
        description="List of detected objects with class names, bounding boxes, attributes, and spatial relationships"
    )
    ocr_text: str = Field(
        default="",
        description="All on-screen text, scoreboards, player names, scores, statistics, jersey names, or slide text read verbatim"
    )
    scene_description: str = Field(
        default="",
        description="Comprehensive 1-2 sentence description of the visual scene and actions"
    )


class VisionEngine:
    """
    Manages OCR extraction, scene graphs, and multimodal visual analysis of keyframes.
    Uses Google Gemini multimodal vision as primary engine with Tesseract fallback.
    """

    def __init__(self, ocr_conf_threshold: int = config.OCR_CONF_THRESHOLD):
        self.ocr_conf_threshold = ocr_conf_threshold

    @staticmethod
    def encode_image_to_base64(image_path: str, max_dim: int = 512, quality: int = 80) -> Optional[str]:
        """
        Encodes an image file to a compact base64 JPEG string for multimodal LLM consumption.
        Downscales oversized frames (e.g. 1080p/4K) to `max_dim` (512px), reducing payload
        RAM footprint by over 95% while retaining crisp visual details for OCR and scoreboards.
        """
        if not os.path.exists(image_path):
            return None
        try:
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                w, h = img.size
                if max(w, h) > max_dim:
                    scale = max_dim / max(w, h)
                    new_size = (int(w * scale), int(h * scale))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                import io
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality, optimize=True)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            # Fallback to direct file read if PIL resize encounters an issue
            try:
                with open(image_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                print(f"Warning: Failed to encode image {image_path}: {e}")
                return None

    def perform_ocr(self, image_path: str) -> List[OCRToken]:
        """
        Extracts on-screen text using Tesseract OCR with adaptive Otsu thresholding.
        
        Args:
            image_path: Path to the keyframe image
            
        Returns:
            List of OCRToken items filtered by confidence threshold
        """
        if not os.path.exists(image_path):
            return []

        # Read image with OpenCV
        image = cv2.imread(image_path)
        if image is None:
            return []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding for high text-background contrast
        processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Run Tesseract with structured data output
        custom_config = r'--oem 3 --psm 6 -c min_orientation_margin=4'
        try:
            ocr_dict = pytesseract.image_to_data(
                processed,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
        except Exception as e:
            # If Tesseract binary is not installed on system PATH, fail gracefully
            print(f"Notice: Tesseract OCR call failed ({e}). Proceeding without OCR text.")
            return []

        tokens: List[OCRToken] = []
        num_boxes = len(ocr_dict.get('text', []))

        for i in range(num_boxes):
            word = ocr_dict['text'][i].strip()
            if not word:
                continue

            try:
                conf = float(ocr_dict['conf'][i])
            except (ValueError, TypeError):
                continue

            if conf >= self.ocr_conf_threshold:
                tokens.append(
                    OCRToken(
                        text=word,
                        confidence=conf,
                        x=int(ocr_dict['left'][i]),
                        y=int(ocr_dict['top'][i]),
                        width=int(ocr_dict['width'][i]),
                        height=int(ocr_dict['height'][i])
                    )
                )

        return tokens

    def inspect_frame_visuals(
        self,
        image_path: str,
        user_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyzes a keyframe using the configured multimodal LLM.
        Extracts objects, spatial relations, and scene descriptions using structured outputs.
        
        Args:
            image_path: Path to keyframe JPEG
            user_prompt: Optional custom question to focus visual attention
            
        Returns:
            Dictionary containing detected objects and scene description
        """
        if not os.path.exists(image_path):
            return {"objects": [], "scene_description": "Image not found."}

        # Encode image to base64 for LLM input
        with open(image_path, "rb") as img_file:
            b64_data = base64.b64encode(img_file.read()).decode("utf-8")

        prompt_instruction = user_prompt or (
            "Analyze this video frame in detail:\n"
            "1. Read and transcribe all on-screen graphics, scoreboards, player names, scores, statistics, jersey names, subtitles, or presentation text verbatim into ocr_text.\n"
            "2. Identify prominent visual objects, their attributes (colors, actions, uniforms), and spatial relationships (above, next_to, inside).\n"
            "3. Provide a concise 1-2 sentence description of the visual scene."
        )

        try:
            # Instantiate multimodal chat model from factory
            llm = LLMFactory.get_chat_model(temperature=0.1)

            # Build multimodal message with image data
            from langchain_core.messages import HumanMessage
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt_instruction},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
                    }
                ]
            )

            # Use with_structured_output if supported, or direct generation
            try:
                structured_llm = llm.with_structured_output(VisionDetectionSchema)
                result: VisionDetectionSchema = structured_llm.invoke([message])
                return {
                    "objects": [obj.model_dump() for obj in result.objects],
                    "ocr_text": result.ocr_text,
                    "scene_description": result.scene_description
                }
            except Exception:
                # Fallback to direct text prompt if provider doesn't support structured schema
                resp = llm.invoke([message])
                return {
                    "objects": [],
                    "ocr_text": "",
                    "scene_description": extract_text_from_message_content(resp.content)
                }

        except Exception as e:
            print(f"Warning: Vision analysis failed on {image_path}: {e}")
            return {"objects": [], "scene_description": f"Visual analysis unavailable ({e})"}
