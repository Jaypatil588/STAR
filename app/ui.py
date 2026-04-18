from __future__ import annotations


def get_ui_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>STAR Stream UI</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/github-dark.min.css" />
  <style>
    :root {
      --bg: #000000;
      --panel: #121212;
      --panel-2: #1a1a1a;
      --border: #2b2b2b;
      --text: #e0e0e0;
      --muted: #9e9e9e;
      --accent: #bdbdbd;
      --ok: #81c784;
      --err: #ef9a9a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at 20% 10%, #161616 0%, #000000 50%);
      color: var(--text);
      font-family: "SF Mono", Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      min-height: 100vh;
    }
    .wrap { max-width: 1280px; margin: 24px auto; padding: 0 16px; }
    .title { font-size: 22px; margin: 0 0 12px; letter-spacing: 0.3px; }
    .controls {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      margin-bottom: 12px;
      display: grid;
      grid-template-columns: 220px 1fr auto;
      gap: 10px;
      align-items: center;
    }
    input, textarea, button {
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      font: inherit;
      padding: 10px;
    }
    textarea { min-height: 88px; resize: vertical; grid-column: 1 / -1; }
    button {
      cursor: pointer;
      background: #2f2f2f;
      color: #f5f5f5;
      font-weight: 700;
      border: 1px solid #4a4a4a;
      height: 42px;
      min-width: 150px;
    }
    button:hover {
      background: #3a3a3a;
    }
    .content { display: grid; grid-template-columns: 70% 30%; gap: 12px; }
    .panel {
      border: 1px solid var(--border);
      border-radius: 12px;
      background: rgba(18, 18, 18, 0.96);
      min-height: 520px;
      overflow: hidden;
    }
    .panel h3 {
      margin: 0;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      font-size: 14px;
      color: var(--muted);
      letter-spacing: 0.2px;
    }
    pre {
      margin: 0;
      padding: 12px;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.4;
      font-size: 13px;
      max-height: 700px;
      overflow: auto;
    }
    .rendered {
      margin: 0;
      padding: 12px;
      line-height: 1.5;
      font-size: 14px;
      max-height: 700px;
      overflow: auto;
    }
    .rendered h2, .rendered h3 {
      margin: 12px 0 8px;
      color: #cfcfcf;
    }
    .rendered p, .rendered ul, .rendered ol {
      margin: 8px 0;
    }
    .rendered code {
      background: #0d0d0d;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 2px 5px;
      font-size: 13px;
    }
    .rendered pre code {
      border: none;
      padding: 0;
      background: transparent;
      font-size: 13px;
    }
    #taskStack {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .task-card {
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #101010;
      padding: 10px;
    }
    .task-title {
      font-size: 13px;
      color: #cfcfcf;
      margin-bottom: 8px;
    }
    .task-status {
      color: var(--muted);
      font-size: 12px;
      margin-left: 8px;
    }
    .hidden { display: none; }
    .status-ok { color: var(--ok); }
    .status-err { color: var(--err); }
    @media (max-width: 980px) {
      .controls { grid-template-columns: 1fr; }
      .content { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">STAR Routing Stream Console</h1>
    <div class="controls">
      <input id="sessionId" placeholder="session_id" value="ui-session-1" />
      <button id="runBtn">Run Prompt</button>
      <div></div>
      <textarea id="prompt" placeholder="Type prompt...">write a creative story and make code game for it</textarea>
    </div>
    <div class="content">
      <section class="panel">
        <h3>Streaming Output</h3>
        <pre id="outputRaw"></pre>
        <div id="outputRendered" class="rendered hidden">
          <div id="taskStack"></div>
        </div>
      </section>
      <section class="panel">
        <h3>Metadata</h3>
        <pre id="meta"></pre>
      </section>
    </div>
  </div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/15.0.12/marked.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/highlight.min.js"></script>
  <script>
    const outputRaw = document.getElementById("outputRaw");
    const outputRendered = document.getElementById("outputRendered");
    const taskStack = document.getElementById("taskStack");
    const meta = document.getElementById("meta");
    const runBtn = document.getElementById("runBtn");
    const promptEl = document.getElementById("prompt");
    const sessionEl = document.getElementById("sessionId");

    function uid() {
      if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
      return "req-" + Date.now();
    }

    function ms3(value) {
      return Number(Number(value).toFixed(3));
    }

    function renderMeta(state) {
      const stackSummary = (state.stack || []).map((task) => ({
        index: task.index,
        tool: task.tool,
        model: task.model,
        done: task.done,
        inference_ms: task.inference_ms,
        output_chars: (task.output || "").length
      }));
      const safeState = {
        ...state,
        stack: stackSummary
      };
      meta.textContent = JSON.stringify(safeState, null, 2);
    }

    function finalizeActiveTask(state, nowMs) {
      if (!state.active_task || state.active_task.done) return;
      state.active_task.done = true;
      state.active_task.inference_ms = ms3(nowMs - state.active_task.started_at_ms);
      state.timing_ms.model_inference_ms[String(state.active_task.index)] = state.active_task.inference_ms;
    }

    function renderTaskStack(state) {
      const blocks = [];
      for (const task of state.stack) {
        const status = task.done ? "done" : "running";
        const duration = task.inference_ms != null
          ? (task.done ? (" completed in " + ms3(task.inference_ms) + " ms") : (" elapsed " + ms3(task.inference_ms) + " ms"))
          : "";
        const body = marked.parse(task.output || "", { breaks: true, gfm: true });
        blocks.push(
          "<article class='task-card'>" +
            "<div class='task-title'>Task " + task.index + ": " + task.tool +
              "<span class='task-status'>[" + task.model + " | " + status + "]" + duration + "</span></div>" +
            "<div>" + body + "</div>" +
          "</article>"
        );
      }
      taskStack.innerHTML = blocks.join("");
      taskStack.querySelectorAll("pre code").forEach((block) => hljs.highlightElement(block));
    }

    function parseLine(line, state) {
      const now = performance.now();
      const header = line.match(/\\[STAR\\] request_id=([^\\s]+) session=([^\\s]+)/);
      if (header) {
        state.request_id = header[1];
        state.session_id = header[2];
        return;
      }
      const timingSlm = line.match(/^\\[TIMING\\]\\s+slm_inference_ms=([0-9.]+)$/);
      if (timingSlm) {
        state.timing_ms.slm_inference_ms = ms3(parseFloat(timingSlm[1]));
        return;
      }
      const timingTask = line.match(/^\\[TIMING\\]\\s+task_index=(\\d+)\\s+model_inference_ms=([0-9.]+)$/);
      if (timingTask) {
        const taskIndex = Number(timingTask[1]);
        const taskMs = ms3(parseFloat(timingTask[2]));
        state.timing_ms.model_inference_ms[String(taskIndex)] = taskMs;
        const card = state.stack.find((t) => t.index === taskIndex);
        if (card) {
          card.inference_ms = taskMs;
          card.done = true;
        }
        return;
      }
      const timingTotal = line.match(/^\\[TIMING\\]\\s+total_ms=([0-9.]+)$/);
      if (timingTotal) {
        state.timing_ms.total_ms = ms3(parseFloat(timingTotal[1]));
        return;
      }
      const split = line.match(/\\[STAR\\] split → (\\d+) task\\(s\\)/);
      if (split) {
        state.split = true;
        state.task_count = Number(split[1]);
        return;
      }
      const single = line.match(/\\[STAR\\] single task → (\\d+) task\\(s\\)/);
      if (single) {
        state.split = false;
        state.task_count = Number(single[1]);
        return;
      }
      const task = line.match(/--- Task (\\d+): (.+) \\(via (.+)\\) ---/);
      if (task) {
        finalizeActiveTask(state, now);
        const card = {
          index: Number(task[1]),
          tool: task[2],
          model: task[3],
          output: "",
          done: false,
          started_at_ms: now,
          inference_ms: null,
        };
        state.stack.unshift(card); // FILO stack: latest at top
        state.active_task = card;
        state.tasks.push({ index: card.index, tool: card.tool, model: card.model });
        return;
      }
      const done = line.match(/\\[STAR\\] Done\\. (\\d+) task\\(s\\) completed\\./);
      if (done) {
        state.tasks_completed = Number(done[1]);
        finalizeActiveTask(state, now);
        return;
      }
      if (state.active_task) {
        state.active_task.output += (state.active_task.output ? "\\n" : "") + line;
      }
    }

    async function runPrompt() {
      const prompt = promptEl.value.trim();
      const session_id = sessionEl.value.trim();
      if (!prompt || !session_id) return;

      const request_id = uid();
      const state = {
        endpoint: "/v1/star",
        started_at: new Date().toISOString(),
        started_at_ms: performance.now(),
        request_id,
        session_id,
        split: null,
        task_count: 0,
        tasks_completed: 0,
        tasks: [],
        stack: [],
        active_task: null,
        timing_ms: {
          slm_inference_ms: null,
          model_inference_ms: {},
          total_ms: null
        },
        status: "running"
      };
      outputRaw.textContent = "";
      outputRaw.classList.remove("hidden");
      outputRendered.classList.add("hidden");
      taskStack.innerHTML = "";
      renderMeta(state);
      runBtn.disabled = true;
      runBtn.textContent = "Running...";

      try {
        const resp = await fetch("/v1/star", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ request_id, session_id, prompt, context: {} })
        });
        if (!resp.ok) {
          throw new Error("HTTP " + resp.status);
        }
        const payload = await resp.json();
        if (payload.request_id) state.request_id = payload.request_id;
        const finalResponse = payload.final_response || {};
        const outputText = (finalResponse && finalResponse.output) ? String(finalResponse.output) : "";

        outputRaw.textContent = outputText || JSON.stringify(payload, null, 2);
        outputRaw.scrollTop = outputRaw.scrollHeight;

        const lines = outputRaw.textContent.split("\\n");
        for (const line of lines) parseLine(line, state);
        renderTaskStack(state);
        outputRaw.classList.add("hidden");
        outputRendered.classList.remove("hidden");
        if (state.timing_ms.total_ms == null) {
          state.timing_ms.total_ms = ms3(performance.now() - state.started_at_ms);
        }
        state.status = "success";
      } catch (err) {
        state.status = "error";
        state.error = String(err);
        state.timing_ms.total_ms = ms3(performance.now() - state.started_at_ms);
      } finally {
        state.finished_at = new Date().toISOString();
        renderMeta(state);
        runBtn.disabled = false;
        runBtn.textContent = "Run Prompt";
      }
    }

    runBtn.addEventListener("click", runPrompt);
  </script>
</body>
</html>
"""
