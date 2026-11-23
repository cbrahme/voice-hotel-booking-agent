"use strict";

/*
 * Voice booking page.
 *
 *   mic ─► SpeechRecognition ─► POST /api/chat ─► reply ─► speechSynthesis
 *
 * States: idle → listening → thinking → speaking → idle.
 * - The mic is never listening while the agent speaks (it would hear itself).
 * - Tapping the mic while the agent speaks cuts it off and starts listening (barge-in).
 * - Replies are spoken one sentence at a time: Chrome stops long utterances after ~15s.
 */

const STEPS = ["dates", "guests", "room", "contact", "awaiting_confirmation", "confirmed"];
const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const synth = "speechSynthesis" in window ? window.speechSynthesis : null;

const $ = (id) => document.getElementById(id);
const ui = {
  body: document.body,
  transcript: $("transcript"),
  mic: $("mic"),
  status: $("status"),
  interim: $("interim"),
  form: $("text-form"),
  input: $("text-input"),
  send: $("send"),
  speakToggle: $("speak-toggle"),
  handsFree: $("handsfree-toggle"),
  newConversation: $("new-conversation"),
  steps: $("steps"),
  banner: $("confirmed-banner"),
  reference: $("reference"),
  rooms: $("rooms"),
  roomsNote: $("rooms-note"),
};

let sessionId = null;
let state = "idle"; // idle | listening | thinking | speaking
let busy = false; // waiting for the server
let complete = false; // booking finished
let lastBooking = {};
let lastRoomsKey = null;

// ------------------------------------------------------------------ preferences

function loadPref(key, fallback) {
  try {
    const v = localStorage.getItem(key);
    return v === null ? fallback : v === "1";
  } catch {
    return fallback;
  }
}

function savePref(key, value) {
  try {
    localStorage.setItem(key, value ? "1" : "0");
  } catch {
    /* storage unavailable: preference just isn't remembered */
  }
}

// ------------------------------------------------------------------ status

function defaultStatus() {
  if (!recognition) return "Type your answer below.";
  return {
    idle: "Tap the mic and start talking",
    listening: "Listening… I'll send when you pause, or tap to send now",
    thinking: "Thinking…",
    speaking: "Speaking… tap the mic to interrupt",
  }[state];
}

function setState(next, message) {
  state = next;
  ui.body.dataset.state = next;
  ui.mic.setAttribute("aria-pressed", String(next === "listening"));
  ui.mic.setAttribute(
    "aria-label",
    next === "listening" ? "Stop listening" : next === "speaking" ? "Interrupt and speak" : "Start speaking",
  );
  ui.mic.disabled = !recognition || next === "thinking";
  ui.status.textContent = message || defaultStatus();
}

function setInputEnabled(enabled) {
  ui.input.disabled = !enabled;
  ui.send.disabled = !enabled;
}

// ------------------------------------------------------------------ transcript

function addMessage(role, text, ms) {
  const row = document.createElement("div");
  row.className = `msg ${role}`;
  const bubble = document.createElement("p");
  bubble.className = "bubble";
  bubble.textContent = text;
  row.append(bubble);
  if (ms != null) {
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = `${(ms / 1000).toFixed(1)}s`;
    meta.title = "Time to get this reply from the server";
    row.append(meta);
  }
  ui.transcript.append(row);
  ui.transcript.scrollTop = ui.transcript.scrollHeight;
}

// ------------------------------------------------------------------ server

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const data = await res.json();
      if (typeof data.detail === "string") detail = data.detail;
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return res.json();
}

function newSessionId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

async function startConversation() {
  stopListening(true);
  stopSpeaking();
  sessionId = newSessionId();
  complete = false;
  busy = false;
  ui.transcript.replaceChildren();
  renderBooking(null);
  setInputEnabled(true);
  try {
    const data = await api("/api/reset", { session_id: sessionId });
    addMessage("agent", data.reply_text);
  } catch {
    addMessage("error", "Can't reach the booking server. Is it running?");
  }
  setState("idle");
}

async function sendMessage(text) {
  text = text.trim();
  if (!text || busy) return;
  stopListening(true);
  stopSpeaking();
  busy = true;
  setInputEnabled(false);
  addMessage("user", text);
  setState("thinking");

  const started = performance.now();
  const session = sessionId;
  let data;
  try {
    data = await api("/api/chat", { session_id: session, user_message: text });
  } catch (err) {
    if (session !== sessionId) return; // user started a new conversation meanwhile
    busy = false;
    setInputEnabled(true);
    addMessage("error", `Something went wrong (${err.message}). Please try again.`);
    setState("idle");
    return;
  }
  if (session !== sessionId) return;
  busy = false;
  setInputEnabled(true);
  addMessage("agent", data.reply_text, performance.now() - started);
  renderBooking(data);
  complete = data.is_complete;
  speak(data.reply_text);
}

// ------------------------------------------------------------------ speaking (TTS)

let voice = null;
let speakToken = 0;
let utterances = []; // keep references: Chrome may drop events for garbage-collected utterances
let speakPoll = null;

function pickVoice() {
  if (!synth) return;
  const english = synth.getVoices().filter((v) => /^en[-_]US/i.test(v.lang));
  const preferred = ["Google US English", "Samantha", "Microsoft Aria", "Microsoft Jenny", "Ava", "Allison"];
  voice =
    english.find((v) => preferred.some((name) => v.name.includes(name))) ||
    english.find((v) => v.localService) ||
    english[0] ||
    null;
}

function splitSentences(text) {
  const parts = text.match(/[^.!?]+[.!?]+["')\]]*|[^.!?]+$/g) || [text];
  return parts.map((p) => p.trim()).filter(Boolean);
}

function speak(text) {
  if (!synth || !ui.speakToggle.checked) {
    doneSpeaking(speakToken);
    return;
  }
  stopSpeaking();
  const token = speakToken;
  setState("speaking");
  utterances = splitSentences(text).map((part) => {
    const u = new SpeechSynthesisUtterance(part);
    u.lang = "en-US";
    u.rate = 1.05;
    if (voice) u.voice = voice;
    u.onerror = (ev) => {
      if (ev.error !== "interrupted" && ev.error !== "canceled") doneSpeaking(token);
    };
    return u;
  });
  let started = false;
  utterances[0].onstart = () => {
    started = true;
  };
  utterances[utterances.length - 1].onend = () => doneSpeaking(token);
  utterances.forEach((u) => synth.speak(u));

  // Safety nets: some browsers occasionally never fire `onend`, and with no usable
  // voice speech may never start at all. Don't leave the page stuck on "Speaking…".
  const queuedAt = performance.now();
  speakPoll = setInterval(() => {
    if (started && !synth.speaking && !synth.pending) doneSpeaking(token);
    else if (!started && performance.now() - queuedAt > 3000) {
      synth.cancel();
      doneSpeaking(token);
    }
  }, 400);
}

function stopSpeaking() {
  speakToken += 1;
  clearInterval(speakPoll);
  utterances = [];
  if (synth && (synth.speaking || synth.pending)) synth.cancel();
}

function doneSpeaking(token) {
  if (token !== speakToken) return; // already finished, or interrupted
  speakToken += 1;
  clearInterval(speakPoll);
  utterances = [];
  setState("idle");
  if (ui.handsFree.checked && !complete) startListening();
}

// ------------------------------------------------------------------ listening (STT)

// Chrome's own end-of-speech detection stops at the slightest pause, which cuts phone
// numbers in half ("555… 123… 4567"). So we listen continuously and decide ourselves
// when the user has finished. Tapping the mic sends immediately.
//
// Chrome marks words "final" a moment after the speaker pauses. We then wait a little
// longer for more speech (much longer while collecting the name and phone number,
// where people pause between digit groups). While words are still provisional we use
// a fallback in case a "final" never arrives.
const AFTER_FINAL_MS = { default: 800, contact: 2000 };
const AFTER_INTERIM_MS = 2000;
const MAX_LISTEN_MS = 30000;

let recognition = null;
let heard = ""; // finalized speech so far
let pending = ""; // the latest not-yet-final words
let discardResult = false;
let silenceTimer = null;
let maxListenTimer = null;
let currentStep = "dates";

function silenceDelay(lastWasFinal) {
  const afterFinal = AFTER_FINAL_MS[currentStep] || AFTER_FINAL_MS.default;
  return lastWasFinal ? afterFinal : Math.max(afterFinal, AFTER_INTERIM_MS);
}

function clearListenTimers() {
  clearTimeout(silenceTimer);
  clearTimeout(maxListenTimer);
}

function joinSpeech(a, b) {
  return `${a} ${b}`.replace(/\s+/g, " ").trim();
}

function micErrorMessage(error) {
  switch (error) {
    case "not-allowed":
    case "service-not-allowed":
      return "Microphone access is blocked. Allow it in the address bar, or type instead.";
    case "no-speech":
      return "I didn't hear anything. Tap the mic to try again.";
    case "audio-capture":
      return "No microphone found. You can type instead.";
    case "network":
      return "Voice input needs an internet connection. You can type instead.";
    default:
      return `Voice input stopped (${error}). Tap the mic to try again.`;
  }
}

function setupRecognition() {
  if (!Recognition) return null;
  const rec = new Recognition();
  rec.lang = "en-US";
  rec.interimResults = true;
  rec.continuous = true; // keep listening through pauses; we decide when to stop
  rec.maxAlternatives = 1;

  rec.onstart = () => {
    heard = "";
    pending = "";
    discardResult = false;
    setState("listening");
    // Safety cap, in case speech never pauses (e.g. a TV in the background).
    maxListenTimer = setTimeout(() => recognition.stop(), MAX_LISTEN_MS);
  };

  rec.onresult = (ev) => {
    pending = "";
    for (let i = ev.resultIndex; i < ev.results.length; i += 1) {
      const result = ev.results[i];
      if (result.isFinal) heard = joinSpeech(heard, result[0].transcript);
      else pending = joinSpeech(pending, result[0].transcript);
    }
    ui.interim.textContent = joinSpeech(heard, pending);
    // Every new word restarts the silence countdown.
    const lastWasFinal = ev.results[ev.results.length - 1].isFinal;
    clearTimeout(silenceTimer);
    silenceTimer = setTimeout(() => recognition.stop(), silenceDelay(lastWasFinal));
  };

  rec.onerror = (ev) => {
    if (ev.error === "aborted") return;
    clearListenTimers();
    if (ev.error === "not-allowed" || ev.error === "service-not-allowed") ui.handsFree.checked = false;
    discardResult = true;
    setState("idle", micErrorMessage(ev.error));
  };

  rec.onend = () => {
    clearListenTimers();
    ui.interim.textContent = "";
    // stop() normally finalizes the last words; fall back to them if it didn't.
    const text = joinSpeech(heard, pending);
    heard = "";
    pending = "";
    if (state === "listening") setState("idle");
    if (text && !discardResult) sendMessage(text);
  };
  return rec;
}

function startListening() {
  if (!recognition || busy || state === "listening") return;
  stopSpeaking();
  try {
    recognition.start();
  } catch {
    /* already started */
  }
}

function stopListening(discard) {
  if (!recognition || state !== "listening") return;
  clearListenTimers();
  if (discard) {
    discardResult = true;
    recognition.abort();
  } else {
    recognition.stop(); // ends and sends what was heard so far
  }
}

// ------------------------------------------------------------------ booking card

const money = (n) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: Number.isInteger(n) ? 0 : 2,
  }).format(n);

function parseIso(iso) {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d);
}

const shortDate = (iso) =>
  parseIso(iso).toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });

function formatPhone(p) {
  const digits = p.replace(/\D/g, "");
  if (digits.length === 10 && !p.startsWith("+")) {
    return `(${digits.slice(0, 3)}) ${digits.slice(3, 6)}-${digits.slice(6)}`;
  }
  return p;
}

function fieldValues(b) {
  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
  let dates = null;
  if (b.check_in_date && b.check_out_date) dates = `${shortDate(b.check_in_date)} → ${shortDate(b.check_out_date)}`;
  else if (b.check_in_date) dates = `${shortDate(b.check_in_date)} → ?`;
  else if (b.check_out_date) dates = `? → ${shortDate(b.check_out_date)}`;
  return {
    dates,
    nights: b.nights ? plural(b.nights, "night") : null,
    guests: b.num_guests ? plural(b.num_guests, "guest") : null,
    room: b.room_type || null,
    name: b.guest_name || null,
    phone: b.phone_number ? formatPhone(b.phone_number) : null,
    total: b.total_price != null ? money(b.total_price) : null,
  };
}

function renderBooking(data) {
  const booking = data ? data.booking_summary : {};
  const step = data ? data.current_step : "dates";
  currentStep = step;

  const current = STEPS.indexOf(step);
  ui.steps.querySelectorAll("li").forEach((li, i) => {
    li.classList.toggle("done", i < current || step === "confirmed");
    li.classList.toggle("current", i === current && step !== "confirmed");
  });

  const before = fieldValues(lastBooking);
  const after = fieldValues(booking);
  document.querySelectorAll("[data-field]").forEach((dd) => {
    const key = dd.dataset.field;
    const value = after[key];
    dd.textContent = value || "—";
    dd.classList.toggle("empty", !value);
    if (value && value !== before[key]) {
      dd.classList.remove("flash");
      void dd.offsetWidth; // restart the animation
      dd.classList.add("flash");
    }
  });

  ui.banner.hidden = !(data && data.is_complete);
  if (data && data.is_complete) ui.reference.textContent = `#${data.reservation_id}`;

  lastBooking = booking;
  refreshRooms(booking, Boolean(data && data.is_complete));
}

async function refreshRooms(booking, isComplete) {
  const hasDates = Boolean(booking.check_in_date && booking.check_out_date);
  // isComplete is part of the key so "rooms left" updates once the booking is made.
  const key = [booking.check_in_date, booking.check_out_date, booking.num_guests, booking.room_type, isComplete]
    .map(String)
    .join("|");
  if (key === lastRoomsKey) return;
  lastRoomsKey = key;

  const params = new URLSearchParams();
  if (hasDates) {
    params.set("check_in", booking.check_in_date);
    params.set("check_out", booking.check_out_date);
  }
  let rooms;
  try {
    const res = await fetch(`/api/inventory?${params}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    rooms = await res.json();
  } catch {
    ui.roomsNote.textContent = "Couldn't load rooms.";
    return;
  }
  if (key !== lastRoomsKey) return; // a newer refresh started meanwhile

  ui.rooms.replaceChildren(
    ...rooms.map((r) => {
      const tooSmall = booking.num_guests && r.max_guests < booking.num_guests;
      const soldOut = hasDates && r.available_count === 0;
      const li = document.createElement("li");
      li.classList.toggle("selected", booking.room_type === r.room_type);
      li.classList.toggle("unavailable", Boolean(tooSmall || soldOut));

      const cell = (cls, text) => {
        const span = document.createElement("span");
        span.className = cls;
        span.textContent = text;
        return span;
      };
      let avail = "";
      if (soldOut) avail = "Sold out";
      else if (tooSmall) avail = "Too small";
      else if (hasDates) avail = `${r.available_count} left`;
      li.append(
        cell("name", r.room_type),
        cell("price", money(r.price_per_night)),
        cell("detail", `Up to ${r.max_guests} guests`),
        cell("avail", avail),
      );
      return li;
    }),
  );
  ui.roomsNote.textContent = hasDates
    ? "Prices are per night. Availability is for your dates."
    : "Prices are per night. Availability shows once you choose dates.";
}

// ------------------------------------------------------------------ wiring

function init() {
  recognition = setupRecognition();

  ui.speakToggle.checked = loadPref("speakReplies", true);
  ui.handsFree.checked = recognition ? loadPref("handsFree", false) : false;
  ui.handsFree.disabled = !recognition;
  ui.speakToggle.disabled = !synth;

  ui.mic.addEventListener("click", () => {
    if (state === "speaking") {
      stopSpeaking(); // barge-in
      setState("idle");
      startListening();
    } else if (state === "listening") {
      stopListening(false);
    } else if (state === "idle") {
      startListening();
    }
  });

  ui.form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const text = ui.input.value;
    ui.input.value = "";
    sendMessage(text);
  });

  ui.speakToggle.addEventListener("change", () => {
    savePref("speakReplies", ui.speakToggle.checked);
    if (!ui.speakToggle.checked && state === "speaking") {
      stopSpeaking();
      setState("idle");
    }
  });
  ui.handsFree.addEventListener("change", () => savePref("handsFree", ui.handsFree.checked));
  ui.newConversation.addEventListener("click", startConversation);

  if (synth) {
    pickVoice();
    if (synth.addEventListener) synth.addEventListener("voiceschanged", pickVoice);
    else synth.onvoiceschanged = pickVoice;
  }

  startConversation().then(() => {
    if (!recognition) {
      setState("idle", "Voice input isn't supported in this browser. Type your answer below.");
    } else if (!window.isSecureContext) {
      setState("idle", "The mic only works on localhost or HTTPS. Type your answer below.");
    }
  });
}

init();
