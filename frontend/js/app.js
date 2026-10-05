/**
 * VidScribe Main Application Orchestrator
 * ---------------------------------------
 * Glues together the video selector, timeline rail, upload workflow,
 * media studio modal, and per-video chat session isolation.
 */

const App = {
  activeVideoId: null,
  videosList: [],
  chatHistoryMap: {}, // Maps video_id -> { html, sessionId, turnsCount }

  async init() {
    // Initialize UI sub-controllers
    Player.init();
    Chat.init();

    // Setup Event Listeners
    this.bindEvents();

    // Fetch initial list of videos
    await this.refreshVideoList();
  },

  bindEvents() {
    // Video selector dropdown change
    const videoSelect = document.getElementById("videoSelect");
    if (videoSelect) {
      videoSelect.addEventListener("change", (e) => {
        this.selectVideo(e.target.value);
      });
    }

    // Upload Modal Toggle
    const btnOpenUpload = document.getElementById("btnOpenUpload");
    const btnCloseUpload = document.getElementById("btnCloseUpload");
    const uploadModal = document.getElementById("uploadModal");

    if (btnOpenUpload && uploadModal) {
      btnOpenUpload.addEventListener("click", () => uploadModal.classList.remove("hidden"));
    }
    if (btnCloseUpload && uploadModal) {
      btnCloseUpload.addEventListener("click", () => uploadModal.classList.add("hidden"));
    }

    // Drag and Drop Upload Handlers
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("videoFileInput");

    if (dropZone && fileInput) {
      dropZone.addEventListener("click", () => fileInput.click());
      
      dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--accent-cyan)";
      });
      
      dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "var(--panel-border)";
      });

      dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--panel-border)";
        if (e.dataTransfer.files.length > 0) {
          this.handleFileUpload(e.dataTransfer.files[0]);
        }
      });

      fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
          this.handleFileUpload(e.target.files[0]);
        }
      });
    }

    // Media Studio Modal Toggle
    const btnOpenStudio = document.getElementById("btnOpenStudio");
    const btnCloseStudio = document.getElementById("btnCloseStudio");
    const studioModal = document.getElementById("studioModal");

    if (btnOpenStudio && studioModal) {
      btnOpenStudio.addEventListener("click", () => studioModal.classList.remove("hidden"));
    }
    if (btnCloseStudio && studioModal) {
      btnCloseStudio.addEventListener("click", () => studioModal.classList.add("hidden"));
    }

    // Media Studio Action Buttons
    const btnGenerateStoryboard = document.getElementById("btnGenerateStoryboard");
    if (btnGenerateStoryboard) {
      btnGenerateStoryboard.addEventListener("click", () => this.generateStoryboard());
    }

    const btnGenerateVideoSummary = document.getElementById("btnGenerateVideoSummary");
    if (btnGenerateVideoSummary) {
      btnGenerateVideoSummary.addEventListener("click", () => this.generateVideoSummary());
    }
  },

  async refreshVideoList() {
    try {
      const data = await API.fetchVideos();
      this.videosList = data.videos || [];

      const selectEl = document.getElementById("videoSelect");
      if (!selectEl) return;

      selectEl.innerHTML = "";
      if (this.videosList.length === 0) {
        selectEl.innerHTML = '<option value="" disabled selected>No videos available (Upload one)</option>';
        this.activeVideoId = null;
        Player.loadVideo("");
        Chat.setChatState("no-video");
        const badgeText = document.getElementById("videoStatusText");
        const badgePill = document.getElementById("videoStatusBadge");
        if (badgeText && badgePill) {
          badgeText.textContent = "No Video Loaded";
          badgePill.className = "status-indicator-pill indexing";
        }
        const countBadge = document.getElementById("sceneCountBadge");
        if (countBadge) countBadge.textContent = "0 Scenes";
        const railEl = document.getElementById("sceneRail");
        if (railEl) {
          railEl.innerHTML = '<div class="empty-scenes-prompt">Upload a video using the button above to detect scenes and generate keyframes.</div>';
        }
        return;
      }

      // Add default prompt option
      const promptOpt = document.createElement("option");
      promptOpt.value = "";
      promptOpt.disabled = true;
      promptOpt.textContent = "Select a video to begin...";
      if (!this.activeVideoId) {
        promptOpt.selected = true;
      }
      selectEl.appendChild(promptOpt);

      this.videosList.forEach((v) => {
        const opt = document.createElement("option");
        opt.value = v.video_id;
        opt.textContent = v.filename;
        if (this.activeVideoId === v.video_id) {
          opt.selected = true;
        }
        selectEl.appendChild(opt);
      });

      // Strict requirement: DO NOT load any video by default
      if (this.activeVideoId) {
        selectEl.value = this.activeVideoId;
      } else {
        this.activeVideoId = null;
        selectEl.value = "";
        Player.loadVideo("");
        Chat.setChatState("no-video");
        const badgeText = document.getElementById("videoStatusText");
        const badgePill = document.getElementById("videoStatusBadge");
        if (badgeText && badgePill) {
          badgeText.textContent = "No Video Selected";
          badgePill.className = "status-indicator-pill ready";
        }
        const countBadge = document.getElementById("sceneCountBadge");
        if (countBadge) countBadge.textContent = "0 Scenes";
        const railEl = document.getElementById("sceneRail");
        if (railEl) {
          railEl.innerHTML = '<div class="empty-scenes-prompt">Select a video from the top bar to inspect detected scenes and keyframes.</div>';
        }
      }
    } catch (err) {
      console.error("Failed to load videos list:", err);
    }
  },

  /**
   * Selects an active video, synchronizes player and timeline,
   * and manages isolated chat window state for that video.
   */
  async selectVideo(videoId, isNewUpload = false) {
    if (!videoId) return;

    // Save previous video's chat window state in local cache
    if (this.activeVideoId && this.activeVideoId !== videoId && Chat.streamEl) {
      this.chatHistoryMap[this.activeVideoId] = {
        html: Chat.streamEl.innerHTML,
        sessionId: Chat.currentSessionId,
        turnsCount: Chat.turnsCount
      };
    }

    this.activeVideoId = videoId;
    const selectEl = document.getElementById("videoSelect");
    if (selectEl) selectEl.value = videoId;

    const videoData = this.videosList.find((v) => v.video_id === videoId);

    // Update status badge
    const badgeText = document.getElementById("videoStatusText");
    const badgePill = document.getElementById("videoStatusBadge");
    if (badgeText && badgePill) {
      if (videoData && videoData.indexed) {
        badgeText.textContent = "Ready to Chat";
        badgePill.className = "status-pill status-ready";
      } else {
        badgeText.textContent = "⚡ Auto-Indexing in background...";
        badgePill.className = "status-pill status-indexing";
      }
    }

    // Load video stream into HTML5 player
    if (videoData) {
      Player.loadVideo(videoData.video_url);
    }

    // Update Media Studio buttons
    const btnPdf = document.getElementById("btnDownloadPdf");
    if (btnPdf) {
      if (videoData && videoData.has_storyboard) {
        btnPdf.href = `/api/media/storyboard/${videoId}`;
        btnPdf.classList.remove("hidden");
      } else {
        btnPdf.classList.add("hidden");
      }
    }

    const btnWatch = document.getElementById("btnWatchVideoSummary");
    if (btnWatch) {
      if (videoData && videoData.has_summary_video) {
        btnWatch.href = `/api/media/summary-video/${videoId}`;
        btnWatch.classList.remove("hidden");
      } else {
        btnWatch.classList.add("hidden");
      }
    }

    // Load scene timeline
    await this.loadTimeline(videoId);

    // Chat Window Management:
    // If it's a new upload or video hasn't been chatted with yet, open brand new chat window
    if (isNewUpload || !this.chatHistoryMap[videoId]) {
      Chat.openNewChatWindow(videoId, videoData?.filename);
    } else {
      // Restore existing conversation window for this video
      const saved = this.chatHistoryMap[videoId];
      Chat.streamEl.innerHTML = saved.html;
      Chat.currentSessionId = saved.sessionId;
      Chat.turnsCount = saved.turnsCount;
      Chat.updateMemoryBadge();
      Chat.scrollToBottom();
    }

    // Automatic Implicit Indexing: User does not need to manually click index
    if (videoData && !videoData.indexed) {
      Chat.setChatState("indexing");
      this.runTier1Indexing(videoId);
    } else if (videoData && videoData.indexed) {
      Chat.setChatState("idle");
    }
  },

  async loadTimeline(videoId) {
    const railEl = document.getElementById("sceneRail");
    const countBadge = document.getElementById("sceneCountBadge");
    if (!railEl) return;

    railEl.innerHTML = '<div class="empty-rail-notice">Loading scene data...</div>';

    try {
      const data = await API.fetchTimeline(videoId);
      const scenes = data.scenes || [];

      if (countBadge) {
        countBadge.textContent = `${scenes.length} Scenes`;
      }

      if (scenes.length === 0) {
        railEl.innerHTML = `
          <div class="empty-rail-notice">
            ⚡ Preparing video scenes & keyframes in the background... Timeline cards will appear automatically.
          </div>
        `;
        return;
      }

      railEl.innerHTML = "";
      scenes.forEach((s) => {
        const card = document.createElement("div");
        card.className = "scene-timeline-card";
        card.title = `Click to seek to Scene ${s.scene_index + 1} (${s.formatted_start})`;

        card.innerHTML = `
          <div class="scene-card-top">
            <span class="scene-title-badge">Scene ${s.scene_index + 1}</span>
            <span class="scene-timecode-badge">${s.formatted_start} - ${s.formatted_end}</span>
          </div>
          <div class="scene-description-text">${this.escapeHtml(s.description || "Video segment")}</div>
        `;

        card.addEventListener("click", () => {
          Player.seekTo(s.start_time);
        });

        railEl.appendChild(card);
      });

    } catch (err) {
      railEl.innerHTML = `<div class="empty-scenes-prompt">Failed to load timeline: ${this.escapeHtml(err.message)}</div>`;
    }
  },

  async runTier1Indexing(targetVideoId) {
    const videoId = targetVideoId || this.activeVideoId;
    if (!videoId) return;

    Chat.setChatState("indexing");
    Chat.addTraceLog(`[INGESTION] ⚡ Auto-Indexing: Performing background Tier 1 indexing for '${videoId}'...`);
    const badgeText = document.getElementById("videoStatusText");
    const badgePill = document.getElementById("videoStatusBadge");
    if (badgeText && badgePill) {
      badgeText.textContent = "⚡ Auto-Indexing in background...";
      badgePill.className = "status-pill status-indexing";
    }

    try {
      const res = await API.processVideo(videoId);
      Chat.addTraceLog(`[INGESTION] ✅ ${res.message || "Auto-indexing complete!"}`);
      
      await this.refreshVideoList();
      if (this.activeVideoId === videoId) {
        await this.loadTimeline(videoId);
        if (badgeText && badgePill) {
          badgeText.textContent = "Ready to Chat";
          badgePill.className = "status-pill status-ready";
        }
      }
    } catch (err) {
      Chat.addTraceLog(`[INGESTION] ❌ Auto-indexing error: ${err.message}`);
    } finally {
      if (this.activeVideoId === videoId) {
        Chat.setChatState("idle");
      }
    }
  },

  /**
   * Requirement 2: When a new video is uploaded, a new chat window is opened.
   */
  async handleFileUpload(file) {
    const progressContainer = document.getElementById("uploadProgress");
    const progressBar = document.getElementById("uploadProgressBar");
    const progressText = document.getElementById("uploadProgressText");

    if (progressContainer) progressContainer.classList.remove("hidden");

    try {
      Chat.addTraceLog(`[INGESTION] Uploading file: ${file.name}...`);
      const uploadRes = await API.uploadVideo(file);
      Chat.addTraceLog(`[INGESTION] Upload successful! Video ID: ${uploadRes.video_id}`);

      // Close upload modal
      const modal = document.getElementById("uploadModal");
      if (modal) modal.classList.add("hidden");

      // Refresh list, select the new video, and explicitly open a brand new chat window
      await this.refreshVideoList();
      await this.selectVideo(uploadRes.video_id, true);

    } catch (err) {
      alert(`Upload failed: ${err.message}`);
    } finally {
      if (progressContainer) progressContainer.classList.add("hidden");
    }
  },

  async generateStoryboard() {
    if (!this.activeVideoId) return;
    const btn = document.getElementById("btnGenerateStoryboard");
    if (btn) {
      btn.textContent = "Generating PDF...";
      btn.disabled = true;
    }

    try {
      Chat.addTraceLog("[MEDIA] Generating Storyboard PDF with scene keyframes...");
      const res = await API.summarizeVideo(this.activeVideoId, false);
      Chat.addTraceLog("[MEDIA] Storyboard PDF generated successfully!");

      const btnPdf = document.getElementById("btnDownloadPdf");
      if (btnPdf && res.storyboard_pdf_url) {
        btnPdf.href = res.storyboard_pdf_url;
        btnPdf.classList.remove("hidden");
      }
    } catch (err) {
      Chat.addTraceLog(`[MEDIA] Storyboard error: ${err.message}`);
    } finally {
      if (btn) {
        btn.textContent = "Generate Storyboard";
        btn.disabled = false;
      }
    }
  },

  async generateVideoSummary() {
    if (!this.activeVideoId) return;
    const btn = document.getElementById("btnGenerateVideoSummary");
    if (btn) {
      btn.textContent = "Compiling Montage...";
      btn.disabled = true;
    }

    try {
      Chat.addTraceLog("[MEDIA] Compiling summary video montage with voiceover narration...");
      const res = await API.summarizeVideo(this.activeVideoId, true);
      Chat.addTraceLog("[MEDIA] Summary video montage generated successfully!");

      const btnWatch = document.getElementById("btnWatchVideoSummary");
      if (btnWatch && res.summary_video_url) {
        btnWatch.href = res.summary_video_url;
        btnWatch.classList.remove("hidden");
      }
    } catch (err) {
      Chat.addTraceLog(`[MEDIA] Video summary error: ${err.message}`);
    } finally {
      if (btn) {
        btn.textContent = "Compile Highlights";
        btn.disabled = false;
      }
    }
  }
};

// Initialize Application once DOM content is ready
document.addEventListener("DOMContentLoaded", () => {
  App.init();
});
