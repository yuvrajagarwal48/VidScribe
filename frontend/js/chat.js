/**
 * VidScribe Agentic Chat Controller
 * ---------------------------------
 * Manages streaming conversational interactions, multi-turn conversational memory,
 * dynamic multi-agent orchestrator dispatch board, and dedicated per-video chat windows.
 */

const Chat = {
  streamEl: null,
  inputEl: null,
  formEl: null,
  traceAccordion: null,
  traceSummaryText: null,
  traceLogsEl: null,
  orchestratorStatusBadge: null,
  chatMemoryText: null,
  currentSocket: null,
  currentSessionId: null,
  turnsCount: 0,
  isProcessing: false,
  isIndexing: false,

  init() {
    this.streamEl = document.getElementById("chatMessageStream");
    this.inputEl = document.getElementById("chatInput");
    this.formEl = document.getElementById("chatForm");
    this.traceAccordion = document.querySelector(".trace-accordion");
    this.traceSummaryText = document.getElementById("traceSummaryText");
    this.traceLogsEl = document.getElementById("traceLogs");
    this.orchestratorStatusBadge = document.getElementById("orchestratorStatusBadge");
    this.chatMemoryText = document.getElementById("chatMemoryText");

    // Form submission listener
    if (this.formEl) {
      this.formEl.addEventListener("submit", (e) => {
        e.preventDefault();
        this.submitQuery();
      });
    }

    // Toggle trace accordion
    const toggleTraceBtn = document.getElementById("toggleTraceBtn");
    if (toggleTraceBtn && this.traceAccordion) {
      toggleTraceBtn.addEventListener("click", () => {
        const isExpanded = this.traceAccordion.classList.toggle("expanded");
        toggleTraceBtn.setAttribute("aria-expanded", isExpanded);
        const logContent = document.getElementById("traceLogContainer");
        if (logContent) {
          logContent.classList.toggle("hidden", !isExpanded);
        }
      });
    }

    // New Chat button (fresh chat session per video)
    const btnNewChat = document.getElementById("btnNewChat");
    if (btnNewChat) {
      btnNewChat.addEventListener("click", () => this.startNewChat());
    }

    // Clear chat button
    const clearBtn = document.getElementById("btnClearChat");
    if (clearBtn) {
      clearBtn.addEventListener("click", () => this.clearChat());
    }

    // Delegated click for suggestion pills
    document.addEventListener("click", (e) => {
      const pill = e.target.closest(".pill-suggestion");
      if (pill) {
        const prompt = pill.getAttribute("data-prompt");
        if (prompt && this.inputEl && !this.isProcessing && !this.isIndexing) {
          this.inputEl.value = prompt;
          this.submitQuery();
        }
      }

      // Delegated click for grounded timestamp pills
      const timePill = e.target.closest(".timestamp-pill");
      if (timePill) {
        const seconds = parseFloat(timePill.getAttribute("data-start-time"));
        if (!isNaN(seconds)) {
          Player.seekTo(seconds);
        }
      }
    });

    // Enter key submits (Shift+Enter for newline)
    if (this.inputEl) {
      this.inputEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          this.submitQuery();
        }
      });

      // Auto-resize input textarea as user types
      this.inputEl.addEventListener("input", () => {
        this.inputEl.style.height = "auto";
        this.inputEl.style.height = Math.min(this.inputEl.scrollHeight, 120) + "px";
      });
    }
  },

  getContainer() {
    if (!this.streamEl) return null;
    let inner = this.streamEl.querySelector(".chat-stream-inner");
    if (!inner) {
      inner = document.createElement("div");
      inner.className = "chat-stream-inner";
      this.streamEl.appendChild(inner);
    }
    return inner;
  },

  /**
   * Initializes or opens a brand new chat window for a video.
   * Requirement: When a new video is uploaded or selected, a new chat window is opened.
   */
  openNewChatWindow(videoId, filename = "") {
    this.currentSessionId = `session_${videoId || "video"}_${Date.now()}`;
    this.turnsCount = 0;
    this.updateMemoryBadge();

    // Reset thinking accordion to clean idle state
    this.resetTraceLog();
    this.resetAgentCards();

    const container = this.getContainer();
    if (!container) return;

    const displayFilename = filename || (App.videosList.find(v => v.video_id === videoId)?.filename) || "Selected Video";

    // Clean current message stream with pristine welcome card
    container.innerHTML = `
      <div class="message-row agent">
        <div class="msg-header">
          <span class="sender-name">VidScribe Assistant</span>
          <span class="time-meta">New Session Initialized</span>
        </div>
        <div class="msg-content">
          <p>A fresh chat window has been initialized for <strong>${this.escapeHtml(displayFilename)}</strong>. Ask questions about events, dialogue, people, or text shown in the video. Multi-turn follow-ups are seamlessly remembered!</p>
          
          <div class="starter-suggestions">
            <div class="suggestion-header">Suggested Questions:</div>
            <div class="suggestion-chips">
              <button class="pill-suggestion" data-prompt="What are the key highlights and summary of this video?">
                <span>What are the key highlights and summary of this video?</span>
              </button>
              <button class="pill-suggestion" data-prompt="At what timestamp does the main action occur?">
                <span>At what timestamp does the main action occur?</span>
              </button>
              <button class="pill-suggestion" data-prompt="What text is written on the screen throughout the scenes?">
                <span>What text is written on the screen?</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    `;

    this.scrollToBottom();
  },

  /**
   * Starts a fresh chat session for the current active video.
   */
  startNewChat() {
    if (!App.activeVideoId) {
      alert("Please upload or select a video first.");
      return;
    }
    // Inform backend to clear prior memory for old session
    if (this.currentSessionId) {
      API.clearSession(this.currentSessionId);
    }
    this.openNewChatWindow(App.activeVideoId);
  },

  updateMemoryBadge() {
    if (!this.chatMemoryText) return;
    if (this.turnsCount === 0) {
      this.chatMemoryText.textContent = "Memory: Fresh";
    } else {
      this.chatMemoryText.textContent = `Memory: ${this.turnsCount} Turn${this.turnsCount > 1 ? "s" : ""}`;
    }
  },

  setChatState(state) {
    const sendBtn = document.getElementById("btnSendMessage");
    const input = this.inputEl;
    if (!input || !sendBtn) return;

    if (state === "processing") {
      this.isProcessing = true;
      input.disabled = true;
      sendBtn.disabled = true;
      sendBtn.innerHTML = '<span class="spinner-sm"></span>';
      input.placeholder = "VidScribe reasoning across speech & visual frames...";
    } else if (state === "indexing") {
      this.isIndexing = true;
      input.disabled = true;
      sendBtn.disabled = true;
      sendBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <polyline points="12 6 12 12 16 14"></polyline>
        </svg>
      `;
      input.placeholder = "⚡ Video indexing in progress... Chat unlocks once ready.";
    } else if (state === "no-video") {
      this.isProcessing = false;
      this.isIndexing = false;
      input.disabled = true;
      sendBtn.disabled = true;
      input.placeholder = "Please upload or select a video to begin";
    } else {
      // idle
      this.isProcessing = false;
      this.isIndexing = false;
      input.disabled = false;
      sendBtn.disabled = false;
      sendBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <line x1="22" y1="2" x2="11" y2="13"></line>
          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>
      `;
      input.placeholder = "Ask anything about the video...";
    }
  },

  submitQuery() {
    if (this.isProcessing || this.isIndexing) return;
    if (!App.activeVideoId) {
      alert("Please upload or select a video first.");
      return;
    }

    const query = this.inputEl.value.trim();
    if (!query) return;

    if (!this.currentSessionId) {
      this.currentSessionId = `session_${App.activeVideoId}_${Date.now()}`;
    }

    // Append user message bubble
    this.appendUserMessage(query);
    this.inputEl.value = "";
    this.inputEl.style.height = "auto";

    // Lock input during generation
    this.setChatState("processing");

    // Reset and prepare trace log and agent cards
    this.resetTraceLog();
    this.resetAgentCards();
    this.activateAgent("Supervisor", "Routing");

    const activeVideoId = App.activeVideoId;
    const sessionId = this.currentSessionId;

    // Append agent bubble with visible loading / typing indicator
    const agentBubble = this.createAgentBubble();
    const bodyEl = agentBubble.querySelector(".msg-content");
    const typingDots = agentBubble.querySelector(".typing-dots");
    const contentEl = agentBubble.querySelector(".bubble-content");

    let rawMarkdown = "";
    let hasReceivedFirstToken = false;

    const onComplete = () => {
      this.setChatState("idle");
      this.turnsCount += 1;
      this.updateMemoryBadge();

      if (typingDots) typingDots.remove();
      if (contentEl) {
        contentEl.style.display = "block";
        contentEl.innerHTML = this.renderMarkdown(rawMarkdown);
      }
      this.scrollToBottom();
      this.finalizeAgentCards();
    };

    // Open WebSocket channel for real-time streaming
    try {
      this.currentSocket = API.openChatSocket(
        activeVideoId,
        (data) => {
          if (data.type === "trace") {
            this.addTraceLog(data.message);
          } else if (data.type === "token") {
            if (!hasReceivedFirstToken) {
              hasReceivedFirstToken = true;
              if (typingDots) typingDots.remove();
              if (contentEl) contentEl.style.display = "block";
            }
            rawMarkdown += data.token;
            if (contentEl) {
              contentEl.innerHTML = this.renderMarkdown(rawMarkdown);
            }
            this.scrollToBottom();
          } else if (data.type === "citations") {
            this.renderCitations(bodyEl, data.segments);
          } else if (data.type === "done") {
            this.updateTraceStatus("Answer Grounded with Timestamps", false);
            onComplete();
          } else if (data.type === "error") {
            this.addTraceLog(`Error: ${data.message}`);
            if (!rawMarkdown) {
              rawMarkdown = `⚠️ **Error generating response:** ${data.message}`;
            }
            onComplete();
          }
        },
        (err) => {
          // Fallback to REST API if WebSocket fails
          this.runRestQueryFallback(query, activeVideoId, sessionId, contentEl, typingDots, bodyEl);
        }
      );

      this.currentSocket.onopen = () => {
        this.currentSocket.send(JSON.stringify({ query, session_id: sessionId }));
        this.updateTraceStatus("Supervisor orchestrating specialized agents...", true);
      };

    } catch (e) {
      this.runRestQueryFallback(query, activeVideoId, sessionId, contentEl, typingDots, bodyEl);
    }
  },

  async runRestQueryFallback(query, videoId, sessionId, contentEl, typingDots, bodyEl) {
    this.updateTraceStatus("Orchestrating agents via REST...", true);
    try {
      const res = await API.queryVideo(query, videoId, sessionId);
      if (typingDots) typingDots.remove();
      if (contentEl) {
        contentEl.style.display = "block";
        contentEl.innerHTML = this.renderMarkdown(res.response || "No response received.");
      }

      if (res.reasoning_steps) {
        res.reasoning_steps.forEach((step) => this.addTraceLog(step));
      }
      if (res.video_segments) {
        this.renderCitations(bodyEl, res.video_segments);
      }
      this.updateTraceStatus("Answer Grounded with Timestamps", false);
    } catch (err) {
      if (typingDots) typingDots.remove();
      if (contentEl) {
        contentEl.style.display = "block";
        contentEl.innerHTML = `<p style="color: var(--accent-rose);">Error: ${this.escapeHtml(err.message)}</p>`;
      }
      this.updateTraceStatus("Error", false);
    } finally {
      this.setChatState("idle");
      this.turnsCount += 1;
      this.updateMemoryBadge();
      this.scrollToBottom();
      this.finalizeAgentCards();
    }
  },

  appendUserMessage(text) {
    const bubble = document.createElement("div");
    bubble.className = "message-row user";
    bubble.innerHTML = `
      <div class="msg-header" style="justify-content: flex-end;">
        <span class="sender-name">You</span>
        <span class="time-meta">• Just now</span>
      </div>
      <div class="msg-content">
        <p>${this.escapeHtml(text)}</p>
      </div>
    `;
    this.getContainer().appendChild(bubble);
    this.scrollToBottom();
  },

  createAgentBubble() {
    const bubble = document.createElement("div");
    bubble.className = "message-row agent";
    bubble.innerHTML = `
      <div class="msg-header">
        <span class="sender-name">VidScribe Assistant</span>
        <span class="time-meta">• Just now</span>
      </div>
      <div class="msg-content">
        <div class="typing-dots">
          <span class="dot"></span>
          <span class="dot"></span>
          <span class="dot"></span>
          <span style="margin-left: 6px; font-size: 0.75rem; color: var(--text-dim);">Analyzing video transcripts & keyframes...</span>
        </div>
        <div class="bubble-content markdown-content" style="display: none;"></div>
      </div>
    `;
    this.getContainer().appendChild(bubble);
    this.scrollToBottom();
    return bubble;
  },

  renderMarkdown(text) {
    if (!text) return "";
    let html = "";
    if (typeof marked !== "undefined" && marked.parse) {
      try {
        html = marked.parse(text);
      } catch (e) {
        html = this.fallbackMarkdown(text);
      }
    } else {
      html = this.fallbackMarkdown(text);
    }

    // Convert bracketed timestamps like [01:23] or [01:23 - 01:45] into clickable interactive pills
    html = html.replace(/\[(\d{1,2}:\d{2}(?:\.\d{1,3})?)(?:\s*-\s*(\d{1,2}:\d{2}(?:\.\d{1,3})?))?\]/g, (match, startStr, endStr) => {
      const parts = startStr.split(":").map(Number);
      const totalSec = parts.length === 2 ? parts[0] * 60 + parts[1] : 0;
      const label = endStr ? `${startStr} - ${endStr}` : startStr;
      return `<button class="timestamp-pill" data-start-time="${totalSec}">
        <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
        ${label}
      </button>`;
    });

    return html;
  },

  fallbackMarkdown(text) {
    let escaped = this.escapeHtml(text);
    escaped = escaped.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');
    escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    escaped = escaped.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    escaped = escaped.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    escaped = escaped.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    escaped = escaped.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    escaped = escaped.replace(/^\s*[-*]\s+(.*$)/gim, '<li>$1</li>');
    escaped = escaped.replace(/(<li>.*<\/li>)/gims, '<ul>$1</ul>');
    escaped = escaped.replace(/\n\n+/g, '</p><p>');
    return `<p>${escaped}</p>`;
  },

  renderCitations(containerEl, segments) {
    if (!segments || segments.length === 0) return;
    if (containerEl.querySelector(".citation-group")) return;

    const citationBox = document.createElement("div");
    citationBox.className = "citation-group";

    const label = document.createElement("div");
    label.className = "citation-group-title";
    label.textContent = "Grounding Evidence (click to seek):";
    citationBox.appendChild(label);

    const pillContainer = document.createElement("div");
    pillContainer.style.display = "flex";
    pillContainer.style.flexWrap = "wrap";
    pillContainer.style.gap = "6px";

    segments.forEach((seg) => {
      const pill = document.createElement("button");
      pill.className = "timestamp-pill";
      pill.setAttribute("data-start-time", seg.start_time);
      pill.innerHTML = `
        <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
        ${seg.formatted_start} - ${seg.formatted_end} (Scene ${seg.scene_index + 1})
      `;
      pillContainer.appendChild(pill);
    });

    citationBox.appendChild(pillContainer);
    containerEl.appendChild(citationBox);
    this.scrollToBottom();
  },

  resetAgentCards() {
    const cards = [
      { id: "agentCardSupervisor", defaultBadge: "Idle" },
      { id: "agentCardRag", defaultBadge: "Idle" },
      { id: "agentCardVision", defaultBadge: "Idle" },
      { id: "agentCardAudio", defaultBadge: "Idle" }
    ];

    cards.forEach(c => {
      const el = document.getElementById(c.id);
      if (el) {
        el.className = "agent-pod";
        const badge = el.querySelector(".agent-pod-status");
        if (badge) {
          badge.className = "agent-pod-status idle";
          badge.textContent = c.defaultBadge;
        }
      }
    });

    if (this.orchestratorStatusBadge) {
      this.orchestratorStatusBadge.textContent = "4 Agents Deployed";
      this.orchestratorStatusBadge.style.color = "var(--accent-cyan)";
    }
  },

  activateAgent(agentKey, badgeText = "Active") {
    let cardId = null;
    if (agentKey === "Supervisor") cardId = "agentCardSupervisor";
    else if (agentKey === "RAG") cardId = "agentCardRag";
    else if (agentKey === "Vision") cardId = "agentCardVision";
    else if (agentKey === "Audio") cardId = "agentCardAudio";

    if (cardId) {
      const card = document.getElementById(cardId);
      if (card) {
        card.className = "agent-pod running";
        const badge = card.querySelector(".agent-pod-status");
        if (badge) {
          badge.className = "agent-pod-status active";
          badge.textContent = badgeText;
        }
      }
    }
  },

  finalizeAgentCards() {
    ["agentCardSupervisor", "agentCardRag", "agentCardVision", "agentCardAudio"].forEach(id => {
      const card = document.getElementById(id);
      if (card && card.classList.contains("running")) {
        card.className = "agent-pod success";
        const badge = card.querySelector(".agent-pod-status");
        if (badge) {
          badge.className = "agent-pod-status done";
          badge.textContent = "Done";
        }
      }
    });

    if (this.orchestratorStatusBadge) {
      this.orchestratorStatusBadge.textContent = "Pipeline Completed";
      this.orchestratorStatusBadge.style.color = "var(--accent-emerald)";
    }
  },

  resetTraceLog() {
    if (this.traceLogsEl) {
      this.traceLogsEl.innerHTML = '<span class="log-empty-msg">// Agent thought traces will stream here in real-time when queries are executed...</span>';
    }
    this.updateTraceStatus("Pipeline: Ready", false);
  },

  addTraceLog(message) {
    if (!this.traceLogsEl) return;

    // Remove empty placeholder message
    const emptyNotice = this.traceLogsEl.querySelector(".log-empty-msg");
    if (emptyNotice) emptyNotice.remove();

    const msgUpper = message.toUpperCase();
    if (msgUpper.includes("SUPERVISOR") || msgUpper.includes("CLASSIFIED INTENT")) {
      this.activateAgent("Supervisor", "Routing");
    }
    if (msgUpper.includes("CHROMADB") || msgUpper.includes("[RAG]") || msgUpper.includes("SEARCHING")) {
      this.activateAgent("RAG", "Retrieving");
    }
    if (msgUpper.includes("[VISION]") || msgUpper.includes("KEYFRAME") || msgUpper.includes("OCR")) {
      this.activateAgent("Vision", "Inspecting");
    }
    if (msgUpper.includes("[INGESTION]") || msgUpper.includes("WHISPER") || msgUpper.includes("AUDIO")) {
      this.activateAgent("Audio", "Processing");
    }
    if (msgUpper.includes("[MEMORY]")) {
      this.activateAgent("Supervisor", "Memory Sync");
    }

    const item = document.createElement("div");
    item.className = "trace-line";

    let prefix = "[ORCHESTRATOR]";
    let content = message;
    const match = message.match(/^(\[[^\]]+\])\s*(.*)$/);
    if (match) {
      prefix = match[1];
      content = match[2];
    }

    item.innerHTML = `<span class="trace-tag">${this.escapeHtml(prefix)}</span> <span class="trace-text">${this.escapeHtml(content)}</span>`;
    this.traceLogsEl.appendChild(item);
    this.traceLogsEl.scrollTop = this.traceLogsEl.scrollHeight;

    let cleanSummary = message;
    if (message.startsWith("[")) {
      cleanSummary = message.replace(/^\[[^\]]+\]\s*/, "");
    }
    this.updateTraceStatus(cleanSummary, true);
  },

  updateTraceStatus(statusText, isRunning) {
    if (this.traceSummaryText) {
      this.traceSummaryText.textContent = `Pipeline: ${statusText}`;
    }
    const orb = document.querySelector(".orchestrator-pulse-orb");
    if (orb) {
      if (isRunning) {
        orb.classList.add("active");
      } else {
        orb.classList.remove("active");
      }
    }
  },

  clearChat() {
    const container = this.getContainer();
    if (container) {
      container.innerHTML = `
        <div class="message-row agent">
          <div class="msg-header">
            <span class="sender-name">VidScribe Assistant</span>
            <span class="time-meta">Just now</span>
          </div>
          <div class="msg-content">
            <p>Chat messages cleared. What would you like to investigate next?</p>
          </div>
        </div>
      `;
    }
    this.turnsCount = 0;
    this.updateMemoryBadge();
  },

  scrollToBottom() {
    if (this.streamEl) {
      this.streamEl.scrollTop = this.streamEl.scrollHeight;
    }
  },

  escapeHtml(str) {
    if (!str) return "";
    return str.replace(/[&<>'"]/g, 
      tag => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        "'": '&#39;',
        '"': '&quot;'
      }[tag] || tag)
    );
  }
};
