/**
 * VidScribe Video Player Controller
 * ---------------------------------
 * Synchronizes HTML5 video playback with timeline indicators,
 * tracks playback time, and enables instant seeking to grounded timestamps.
 */

const Player = {
  videoEl: null,
  currentDisplay: null,
  durationDisplay: null,
  placeholderEl: null,

  init() {
    this.videoEl = document.getElementById("mainVideoPlayer");
    this.currentDisplay = document.getElementById("currentTimeDisplay");
    this.durationDisplay = document.getElementById("durationDisplay");
    this.placeholderEl = document.getElementById("playerPlaceholder");

    if (!this.videoEl) return;

    // Track playback time
    this.videoEl.addEventListener("timeupdate", () => this.onTimeUpdate());
    this.videoEl.addEventListener("loadedmetadata", () => this.onLoadedMetadata());
  },

  loadVideo(videoUrl) {
    if (!this.videoEl) return;
    
    if (!videoUrl) {
      if (this.placeholderEl) {
        this.placeholderEl.classList.remove("hidden");
      }
      this.videoEl.removeAttribute("src");
      this.videoEl.load();
      if (this.currentDisplay) this.currentDisplay.textContent = "00:00:00";
      if (this.durationDisplay) this.durationDisplay.textContent = "00:00:00";
      return;
    }

    if (this.placeholderEl) {
      this.placeholderEl.classList.add("hidden");
    }

    this.videoEl.src = videoUrl;
    this.videoEl.load();
  },

  seekTo(seconds) {
    if (!this.videoEl) return;
    this.videoEl.currentTime = Math.max(0, seconds);
    this.videoEl.play().catch(() => {});
  },

  onTimeUpdate() {
    if (!this.videoEl || !this.currentDisplay) return;
    this.currentDisplay.textContent = this.formatTime(this.videoEl.currentTime);
  },

  onLoadedMetadata() {
    if (!this.videoEl || !this.durationDisplay) return;
    this.durationDisplay.textContent = this.formatTime(this.videoEl.duration);
  },

  formatTime(totalSeconds) {
    if (isNaN(totalSeconds)) return "00:00:00";
    const hrs = Math.floor(totalSeconds / 3600);
    const mins = Math.floor((totalSeconds % 3600) / 60);
    const secs = Math.floor(totalSeconds % 60);
    return `${hrs.toString().padStart(2, "0")}:${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  },

  parseTimestampToSeconds(timestampStr) {
    if (!timestampStr) return 0;
    // Format could be HH:MM:SS or MM:SS
    const parts = timestampStr.split(":").map(Number);
    if (parts.length === 3) {
      return parts[0] * 3600 + parts[1] * 60 + parts[2];
    } else if (parts.length === 2) {
      return parts[0] * 60 + parts[1];
    }
    return parseFloat(timestampStr) || 0;
  }
};
