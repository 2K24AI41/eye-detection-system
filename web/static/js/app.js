"use strict";

const video = document.getElementById("video");
const placeholder = document.getElementById("placeholder");
const startButton = document.getElementById("start");
const stopButton = document.getElementById("stop");
const dot = document.getElementById("dot");
const stateText = document.getElementById("state");
const message = document.getElementById("message");
const faces = document.getElementById("faces");
const eyes = document.getElementById("eyes");
const fps = document.getElementById("fps");

const LABELS = { stopped: "Stopped", starting: "Starting...", running: "Running",
                 stopping: "Stopping...", error: "Error", unknown: "Unknown" };
const PLACEHOLDERS = { stopped: "Camera is off", starting: "Starting camera...",
                       stopping: "Stopping camera...", error: "Camera error",
                       unknown: "Server not reachable" };
const MAX_STREAM_RETRIES = 3;

let streaming = false;       // the <img> currently has a stream URL
let streamFailed = false;    // retries used up; wait for the next start before trying again
let streamRetries = 0;
let retryTimer = null;
let lastState = "unknown";
let pending = false;         // a Start/Stop request is in flight
let issued = 0;              // id of the newest request sent
let applied = 0;             // id of the newest response shown

function openStream() {
  video.src = "/video_feed?t=" + Date.now();     // fresh URL so the browser reconnects
  video.hidden = false;
  placeholder.hidden = true;
  streaming = true;
}

function closeStream() {
  clearTimeout(retryTimer);
  retryTimer = null;
  if (streaming) {
    streaming = false;                            // set first: removing src must not look like an error
    video.removeAttribute("src");                 // drops the connection
  }
  video.hidden = true;
  placeholder.hidden = false;
}

video.addEventListener("load", () => { streamRetries = 0; });

video.addEventListener("error", () => {
  if (!streaming) return;
  closeStream();
  if (lastState !== "running") return;
  if (streamRetries >= MAX_STREAM_RETRIES) {
    streamFailed = true;
    placeholder.textContent = "Video stream lost";
    message.className = "message error";
    message.textContent = "The video stream was lost. Press Stop, then Start Camera.";
    return;
  }
  streamRetries += 1;
  placeholder.textContent = "Reconnecting video (" + streamRetries + "/" + MAX_STREAM_RETRIES + ")...";
  retryTimer = setTimeout(() => {
    retryTimer = null;
    if (lastState === "running" && !streamFailed) openStream();
  }, 1000 * streamRetries);
});

function updateButtons(state) {
  const busy = state === "starting" || state === "running" || state === "stopping";
  const reachable = state !== "unknown";
  startButton.disabled = pending || busy || !reachable;
  stopButton.disabled = pending || !busy || state === "stopping" || !reachable;
}

function render(status) {
  const state = status.state;
  lastState = state;
  dot.className = "dot " + state;
  stateText.textContent = LABELS[state] || state;
  faces.textContent = status.faces;
  eyes.textContent = status.eyes;
  fps.textContent = Number(status.fps).toFixed(1);
  updateButtons(state);

  if (state !== "running") {
    streamRetries = 0;
    streamFailed = false;
  }
  if (state === "running") {
    if (!streaming && !streamFailed && retryTimer === null) openStream();
  } else {
    closeStream();
    placeholder.textContent = PLACEHOLDERS[state] || "";
  }

  if (!(state === "running" && streamFailed)) {
    message.className = "message" + (status.error ? " error" : "");
    message.textContent = status.error || status.note || "";
  }
}

function renderUnreachable(text) {
  // Do not keep showing the old status: it may no longer be true.
  lastState = "unknown";
  dot.className = "dot unknown";
  stateText.textContent = LABELS.unknown;
  faces.textContent = eyes.textContent = fps.textContent = "-";
  closeStream();
  placeholder.textContent = PLACEHOLDERS.unknown;
  updateButtons("unknown");
  message.className = "message error";
  message.textContent = text;
}

async function readStatus(response) {
  // Returns {status} on success or {error} with a readable message.
  let body = null;
  try {
    body = await response.json();
  } catch (error) {
    body = null;                                  // not JSON (for example an HTML error page)
  }
  if (!response.ok) {
    const detail = body && typeof body.error === "string" ? body.error : "";
    return { error: "Server error " + response.status + (detail ? ": " + detail : "") };
  }
  if (!body || typeof body.state !== "string") {
    return { error: "The server sent an unexpected reply." };
  }
  return { status: body };
}

async function call(method, url) {
  const id = ++issued;
  let result;
  try {
    result = await readStatus(await fetch(url, { method: method }));
  } catch (error) {
    result = { error: "Cannot reach the server." };
  }
  if (id < applied) return result;                // an older answer must not overwrite a newer one
  applied = id;
  if (result.status) {
    render(result.status);
  } else {
    renderUnreachable(result.error);
  }
  return result;
}

async function action(url) {
  pending = true;                                 // immediate feedback: buttons off at once
  startButton.disabled = true;
  stopButton.disabled = true;
  try {
    await call("POST", url);
  } finally {
    pending = false;
    updateButtons(lastState);
  }
}

startButton.addEventListener("click", () => action("/api/start"));
stopButton.addEventListener("click", () => action("/api/stop"));

call("GET", "/api/status");
setInterval(() => { if (!pending) call("GET", "/api/status"); }, 1000);
