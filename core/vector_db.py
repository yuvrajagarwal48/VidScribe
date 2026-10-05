"""
Vector Database Store
---------------------
ChromaDB persistent vector database manager.
Handles embedding generation, multimodal scene indexing (OCR, ASR, Objects, Combined),
and semantic similarity retrieval with temporal and video-specific metadata filtering.
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import config
from schemas.video_models import SceneAnalysisResult


class VectorDBStore:
    """
    Manages persistent ChromaDB vector storage for multimodal video scenes.
    """

    def __init__(self, persist_dir: str = str(config.VECTOR_DB_DIR)):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Embedding function: Fast ONNX-based all-MiniLM-L6-v2
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            self.embedding_function = DefaultEmbeddingFunction()
        except Exception:
            self.embedding_function = SentenceTransformerEmbeddingFunction(
                model_name=config.EMBEDDING_MODEL_NAME
            )

        # Get or create the unified collection with cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=config.VECTOR_DB_COLLECTION,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def index_scene(self, analysis: SceneAnalysisResult) -> List[str]:
        """
        Splits a scene's multimodal data into indexed documents with rich temporal metadata.
        
        Creates:
        1. An ASR document (if dialogue exists)
        2. An OCR document (if on-screen text exists)
        3. A visual objects document (if visual descriptions exist)
        4. A combined multimodal scene document
        
        Returns:
            List of generated document IDs stored in ChromaDB
        """
        video_id = analysis.video_id
        scene_idx = analysis.scene_index
        doc_ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        base_meta = {
            "video_id": video_id,
            "scene_index": scene_idx,
            "start_time": float(analysis.start_time),
            "end_time": float(analysis.end_time),
            "formatted_start": analysis.formatted_start,
            "formatted_end": analysis.formatted_end,
            "keyframe_path": analysis.keyframe_path
        }

        # 1. Spoken Audio (ASR)
        asr_texts = [seg.transcript for seg in analysis.transcripts if seg.transcript.strip()]
        if asr_texts:
            asr_content = "Spoken Dialogue: " + " ".join(asr_texts)
            doc_id = f"{video_id}_s{scene_idx}_asr"
            doc_ids.append(doc_id)
            documents.append(asr_content)
            metadatas.append({**base_meta, "type": "asr"})

        # 2. On-screen Text (OCR)
        ocr_words = [token.text for token in analysis.ocr_texts if token.text.strip()]
        if ocr_words:
            ocr_content = "On-Screen Text: " + " ".join(ocr_words)
            doc_id = f"{video_id}_s{scene_idx}_ocr"
            doc_ids.append(doc_id)
            documents.append(ocr_content)
            metadatas.append({**base_meta, "type": "ocr"})

        # 3. Visual Objects & Description
        obj_names = [obj.class_name for obj in analysis.detected_objects]
        vis_parts = []
        if analysis.summary_text:
            vis_parts.append(analysis.summary_text)
        if obj_names:
            vis_parts.append("Visible objects: " + ", ".join(obj_names))

        if vis_parts:
            vis_content = "Visual Elements: " + " | ".join(vis_parts)
            doc_id = f"{video_id}_s{scene_idx}_vis"
            doc_ids.append(doc_id)
            documents.append(vis_content)
            metadatas.append({**base_meta, "type": "visual"})

        # 4. Holistically Fused Combined Scene Document
        combined_text = (
            f"Scene {scene_idx} ({analysis.formatted_start} to {analysis.formatted_end}). "
            f"{' '.join(vis_parts)}. "
            f"{' '.join(asr_texts)}. "
            f"{' '.join(ocr_words)}"
        ).strip()

        if combined_text:
            doc_id = f"{video_id}_s{scene_idx}_combined"
            doc_ids.append(doc_id)
            documents.append(combined_text)
            metadatas.append({**base_meta, "type": "combined"})

        # Upsert into ChromaDB
        if doc_ids:
            try:
                self.collection.upsert(
                    ids=doc_ids,
                    documents=documents,
                    metadatas=metadatas
                )
            except Exception as e:
                print(f"Error indexing scene {scene_idx} for video {video_id}: {e}")

        return doc_ids

    def enrich_scene_visuals(
        self,
        video_id: str,
        scene_idx: int,
        scene_description: str,
        ocr_text: str = "",
        objects: Optional[List[Dict[str, Any]]] = None,
        keyframe_path: str = "",
        formatted_start: str = "00:00",
        formatted_end: str = "00:00"
    ):
        """
        Enriches an existing scene in ChromaDB with visual scene descriptions,
        OCR text, and detected object scene graphs.
        """
        doc_ids = []
        documents = []
        metadatas = []

        base_meta = {
            "video_id": video_id,
            "scene_index": scene_idx,
            "formatted_start": formatted_start,
            "formatted_end": formatted_end,
            "keyframe_path": keyframe_path
        }

        # 1. OCR text document
        if ocr_text and ocr_text.strip():
            doc_ids.append(f"{video_id}_s{scene_idx}_ocr")
            documents.append(f"On-Screen Text & Scoreboard: {ocr_text.strip()}")
            metadatas.append({**base_meta, "type": "ocr"})

        # 2. Visual elements & scene graph
        obj_names = []
        rel_texts = []
        if objects:
            for obj in objects:
                c_name = obj.get("class_name", "object")
                attrs = ", ".join(obj.get("attributes", []))
                obj_str = f"{c_name} ({attrs})" if attrs else c_name
                obj_names.append(obj_str)
                for rel in obj.get("relationships", []):
                    rel_texts.append(f"{c_name} {rel.get('relation', 'related_to')} object_{rel.get('target', 0)}")

        vis_parts = []
        if scene_description and scene_description.strip():
            vis_parts.append(scene_description.strip())
        if obj_names:
            vis_parts.append("Objects: " + "; ".join(obj_names))
        if rel_texts:
            vis_parts.append("Scene Graph Relations: " + "; ".join(rel_texts))

        if vis_parts:
            doc_ids.append(f"{video_id}_s{scene_idx}_vis")
            documents.append("Visual Scene & Scene Graph: " + " | ".join(vis_parts))
            metadatas.append({**base_meta, "type": "visual"})

        # 3. Enhanced Combined Document
        combined_text = (
            f"Scene {scene_idx} [{formatted_start} - {formatted_end}]. "
            f"{scene_description}. "
            f"{'On-Screen: ' + ocr_text if ocr_text else ''}. "
            f"{'Objects: ' + ', '.join(obj_names) if obj_names else ''}"
        ).strip()

        if combined_text:
            doc_ids.append(f"{video_id}_s{scene_idx}_combined")
            documents.append(combined_text)
            metadatas.append({**base_meta, "type": "combined"})

        if doc_ids:
            try:
                self.collection.upsert(
                    ids=doc_ids,
                    documents=documents,
                    metadatas=metadatas
                )
            except Exception as e:
                print(f"Error enriching scene {scene_idx} for {video_id}: {e}")

    def query(
        self,
        query_text: str,
        video_id: Optional[str] = None,
        modality_filter: Optional[str] = None,
        top_k: int = config.TOP_K_RESULTS
    ) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB for semantically similar scene chunks.
        
        Args:
            query_text: The search query string
            video_id: Optional filter for a specific video ID
            modality_filter: Optional filter ('asr', 'ocr', 'visual', 'combined')
            top_k: Number of nearest matches to return
            
        Returns:
            List of dictionaries with document text, metadata, and distance
        """
        where_clauses = []
        if video_id:
            where_clauses.append({"video_id": {"$eq": video_id}})
        if modality_filter:
            where_clauses.append({"type": {"$eq": modality_filter}})

        # Build ChromaDB 'where' filter
        where_filter = None
        if len(where_clauses) == 1:
            where_filter = where_clauses[0]
        elif len(where_clauses) > 1:
            where_filter = {"$and": where_clauses}

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where_filter
            )

            hits: List[Dict[str, Any]] = []
            if results and results.get("documents") and results["documents"][0]:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

                for doc, meta, dist in zip(docs, metas, dists):
                    hits.append({
                        "document": doc,
                        "metadata": meta,
                        "distance": dist
                    })

            return hits

        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    def get_all_scenes_for_video(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all combined scene descriptors for a video, ordered by scene_index.
        Used for full-video summarization and timeline generation.
        """
        try:
            results = self.collection.get(
                where={
                    "$and": [
                        {"video_id": {"$eq": video_id}},
                        {"type": {"$eq": "combined"}}
                    ]
                }
            )
            
            scenes = []
            if results and results.get("documents"):
                for doc, meta in zip(results["documents"], results["metadatas"]):
                    scenes.append({
                        "document": doc,
                        "metadata": meta
                    })
                
                # Sort scenes chronologically by scene_index
                scenes.sort(key=lambda s: s["metadata"].get("scene_index", 0))

            return scenes
        except Exception as e:
            print(f"Error fetching scenes for {video_id}: {e}")
            return []

    def get_concluding_scenes_for_video(self, video_id: str, count: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieves the final N scenes of the video (the match conclusion, final scorecard,
        and outcome). Used for temporal anchoring on result/outcome questions.
        """
        scenes = self.get_all_scenes_for_video(video_id)
        if not scenes:
            return []
        return scenes[-count:]
