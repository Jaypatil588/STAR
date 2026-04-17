from __future__ import annotations


def get_ui_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>STAR Stream UI</title>
  <style>
    :root {
      --bg: #0b1220;
      --panel: #111b2f;
      --panel-2: #0f1729;
      --border: #253350;
      --text: #e5edf8;
      --muted: #9db0d1;
      --accent: #4cc9f0;
      --ok: #66d18f;
      --err: #f07178;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at 20% 10%, #13203a 0%, #0b1220 45%);
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
      background: linear-gradient(90deg, #3aa0ff, #4cc9f0);
      color: #041224;
      font-weight: 700;
      border: none;
      height: 42px;
      min-width: 150px;
    }
    .content { display: grid; grid-template-columns: 70% 30%; gap: 12px; }
    .panel {
      border: 1px solid var(--border);
      border-radius: 12px;
      background: rgba(17, 27, 47, 0.92);
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
        <pre id="output"></pre>
      </section>
      <section class="panel">
        <h3>Metadata (30%)</h3>
        <pre id="meta"></pre>
      </section>
    </div>
  </div>

  <script>
    const output = document.getElementById("output");
    const meta = document.getElementById("meta");
    const runBtn = document.getElementById("runBtn");
    const promptEl = document.getElementById("prompt");
    const sessionEl = document.getElementById("sessionId");

    function uid() {
      if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
      return "req-" + Date.now();
    }

    function renderMeta(state) {
      meta.textContent = JSON.stringify(state, null, 2);
    }

    function parseLine(line, state) {
      const header = line.match(/\\[STAR\\] request_id=([^\\s]+) session=([^\\s]+)/);
      if (header) {
        state.request_id = header[1];
        state.session_id = header[2];
      }
      const split = line.match(/\\[STAR\\] split → (\\d+) task\\(s\\)/);
      if (split) {
        state.split = true;
        state.task_count = Number(split[1]);
      }
      const single = line.match(/\\[STAR\\] single task → (\\d+) task\\(s\\)/);
      if (single) {
        state.split = false;
        state.task_count = Number(single[1]);
      }
      const task = line.match(/--- Task (\\d+): (.+) \\(via (.+)\\) ---/);
      if (task) {
        state.tasks.push({ index: Number(task[1]), tool: task[2], model: task[3] });
      }
      const done = line.match(/\\[STAR\\] Done\\. (\\d+) task\\(s\\) completed\\./);
      if (done) {
        state.tasks_completed = Number(done[1]);
      }
    }

    async function runPrompt() {
      const prompt = promptEl.value.trim();
      const session_id = sessionEl.value.trim();
      if (!prompt || !session_id) return;

      const request_id = uid();
      const state = {
        endpoint: "/v1/clean",
        started_at: new Date().toISOString(),
        request_id,
        session_id,
        split: null,
        task_count: 0,
        tasks_completed: 0,
        tasks: [],
        status: "running"
      };
      output.textContent = "";
      renderMeta(state);
      runBtn.disabled = true;
      runBtn.textContent = "Running...";

      try {
        const resp = await fetch("/v1/clean", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ request_id, session_id, prompt })
        });
        if (!resp.ok || !resp.body) {
          throw new Error("HTTP " + resp.status);
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffered = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          output.textContent += chunk;
          output.scrollTop = output.scrollHeight;
          buffered += chunk;
          const lines = buffered.split("\\n");
          buffered = lines.pop() || "";
          for (const line of lines) parseLine(line, state);
          renderMeta(state);
        }
        if (buffered) parseLine(buffered, state);
        state.status = "success";
      } catch (err) {
        state.status = "error";
        state.error = String(err);
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
