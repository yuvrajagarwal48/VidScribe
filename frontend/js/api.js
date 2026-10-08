/**
 * VidScribe API Client Module
 * ----------------------------
 * Handles REST requests and WebSocket streaming communication with the FastAPI backend.
 * Supports multi-turn conversational session memory and timeline retrieval.
 */

const API = {
  baseUrl: window.location.origin,

  /**
   * Fetches the list of all uploaded/indexed videos.
   */
  async fetchVideos() {
    const res = await fetch(`${this.baseUrl}/api/videos`);
    if (!res.ok) throw new Error("Failed to fetch videos");
    return await res.json();
  },

  /**
   * Fetches the scene timeline and keyframe URLs for a specific video.
   */
  async fetchTimeline(videoId) {
    const encId = encodeURIComponent(videoId);
    let res = await fetch(`${this.baseUrl}/api/videos/${encId}/timeline`);
    if (!res.ok) {
      res = await fetch(`${this.baseUrl}/api/timeline/${encId}`);
    }
    if (!res.ok) throw new Error("Failed to fetch timeline");
    return await res.json();
  },

  /**
   * Uploads a video file with multipart form data.
   * Returns video_id and a fresh session_id for a new chat window.
   */
  async uploadVideo(file) {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${this.baseUrl}/api/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("Video upload failed");
    return await res.json();
  },

  /**
   * Triggers progressive Tier 1 indexing on a video.
   */
  async processVideo(videoId) {
    const encId = encodeURIComponent(videoId);
    const res = await fetch(`${this.baseUrl}/api/process/${encId}`, {
      method: "POST",
    });
    if (!res.ok) throw new Error("Video processing failed");
    return await res.json();
  },

  /**
   * Submits a synchronous RAG query over HTTP REST with session memory.
   */
  async queryVideo(query, videoId, sessionId = null) {
    const res = await fetch(`${this.baseUrl}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        query, 
        video_id: videoId,
        session_id: sessionId
      }),
    });
    if (!res.ok) throw new Error("Query execution failed");
    return await res.json();
  },

  /**
   * Resets short-term memory for a given session.
   */
  async clearSession(sessionId) {
    if (!sessionId) return;
    try {
      await fetch(`${this.baseUrl}/api/chat/session/${sessionId}`, { method: "DELETE" });
    } catch (e) {
      console.warn("Failed to clear session on backend:", e);
    }
  },

  /**
   * Triggers video summarization, storyboard generation, or video montage.
   */
  async summarizeVideo(videoId, createVideoClip = false) {
    const res = await fetch(`${this.baseUrl}/api/summarize/${videoId}?create_video_clip=${createVideoClip}`, {
      method: "POST",
    });
    if (!res.ok) throw new Error("Summarization failed");
    return await res.json();
  },

  /**
   * Opens a WebSocket connection for real-time streaming answers and agent thought logs.
   */
  openChatSocket(videoId, onMessage, onError, onClose) {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/chat/${videoId || "all"}`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (e) {
        console.error("Error parsing WebSocket payload:", e);
      }
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      if (onError) onError(err);
    };

    ws.onclose = () => {
      if (onClose) onClose();
    };

    return ws;
  }
};
