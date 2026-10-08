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

        # Embedding function selection:
        # 1. Gemini text-embedding-004 (remote API, 0 MB local RAM, ideal for Render)
        # 2. Fast local ONNX DefaultEmbeddingFunction (no PyTorch, low memory)
        # 3. SentenceTransformers fallback
        embedding_provider = getattr(config, "EMBEDDING_PROVIDER", "auto").lower()
        if embedding_provider == "gemini" and config.GEMINI_API_KEY:
            try:
                from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
                self.embedding_function = GoogleGenerativeAiEmbeddingFunction(
                    api_key=config.GEMINI_API_KEY,
                    model_name="models/text-embedding-004"
                )
                print("VectorDB: Using Google Gemini API text-embedding-004 (zero local RAM).")
            except Exception as e:
                print(f"Notice: GoogleGenerativeAiEmbeddingFunction setup notice ({e}), using default ONNX.")
                self.embedding_function = None
        else:
            self.embedding_function = None

        if self.embedding_function is None:
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

    def _prepare_scene_docs(self, analysis: SceneAnalysisResult) -> tuple:
        """
        Extracts ASR, OCR, Visual, and Combined documents and metadata for a single scene.
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
            "keyframe_path": analysis.keyframe_path or ""
        }

        # 1. Spoken Audio (ASR)
        asr_texts = [seg.transcript for seg in analysis.transcripts if seg.transcript.strip()]
        if asr_texts:
            asr_content = "Spoken Dialogue: " + " ".join(asr_texts)
            doc_ids.append(f"{video_id}_s{scene_idx}_asr")
            documents.append(asr_content)
            metadatas.append({**base_meta, "type": "asr"})

        # 2. On-screen Text (OCR)
        ocr_words = [token.text for token in analysis.ocr_texts if token.text.strip()]
        if ocr_words:
            ocr_content = "On-Screen Text: " + " ".join(ocr_words)
            doc_ids.append(f"{video_id}_s{scene_idx}_ocr")
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
            doc_ids.append(f"{video_id}_s{scene_idx}_vis")
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
            doc_ids.append(f"{video_id}_s{scene_idx}_combined")
            documents.append(combined_text)
            metadatas.append({**base_meta, "type": "combined"})

        return doc_ids, documents, metadatas

    def index_scene(self, analysis: SceneAnalysisResult) -> List[str]:
        """
        Indexes a single scene's multimodal data into ChromaDB.
        """
        doc_ids, documents, metadatas = self._prepare_scene_docs(analysis)
        if doc_ids:
            try:
                self.collection.upsert(
                    ids=doc_ids,
                    documents=documents,
                    metadatas=metadatas
                )
            except Exception as e:
                print(f"Error indexing scene {analysis.scene_index} for video {analysis.video_id}: {e}")
        return doc_ids

    def index_scenes_batch(self, analyses: List[SceneAnalysisResult], batch_size: int = 64) -> List[str]:
        """
        Batches multiple scenes into a single array payload for rapid vector embedding
        and single-transaction SQLite disk commit in ChromaDB.
        Cuts indexing latency from ~18s down to ~2s.
        """
        all_ids: List[str] = []
        all_docs: List[str] = []
        all_metas: List[Dict[str, Any]] = []

        for analysis in analyses:
            d_ids, d_docs, d_metas = self._prepare_scene_docs(analysis)
            all_ids.extend(d_ids)
            all_docs.extend(d_docs)
            all_metas.extend(d_metas)

        if not all_ids:
            return []

        # Batch upsert in array slices
        total_indexed: List[str] = []
        for i in range(0, len(all_ids), batch_size):
            b_ids = all_ids[i:i + batch_size]
            b_docs = all_docs[i:i + batch_size]
            b_metas = all_metas[i:i + batch_size]
            try:
                self.collection.upsert(
                    ids=b_ids,
                    documents=b_docs,
                    metadatas=b_metas
                )
                total_indexed.extend(b_ids)
            except Exception as e:
                print(f"Error in batch vector indexing chunk ({len(b_ids)} items): {e}")

        return total_indexed

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
