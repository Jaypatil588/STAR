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
        <div id="outputRendered" class="rendered hidden"></div>
      </section>
      <section class="panel">
        <h3>Metadata (30%)</h3>
        <pre id="meta"></pre>
      </section>
    </div>
  </div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/15.0.12/marked.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/highlight.min.js"></script>
  <script>
    const outputRaw = document.getElementById("outputRaw");
    const outputRendered = document.getElementById("outputRendered");
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

    function toMarkdown(rawText) {
      const lines = rawText.split("\\n");
      const transformed = [];
      for (const line of lines) {
        const header = line.match(/^\\[STAR\\] request_id=(\\S+) session=(\\S+)/);
        if (header) {
          transformed.push("### Request");
          transformed.push("- request_id: `" + header[1] + "`");
          transformed.push("- session_id: `" + header[2] + "`");
          continue;
        }
        const analyze = line.match(/^\\[STAR\\] Analyzing prompt\\.\\.\\.$/);
        if (analyze) {
          transformed.push("> Analyzing prompt...");
          continue;
        }
        const split = line.match(/^\\[STAR\\] split → (\\d+) task\\(s\\) dispatched concurrently$/);
        if (split) {
          transformed.push("### Routing");
          transformed.push("- split: `true`");
          transformed.push("- task_count: `" + split[1] + "`");
          continue;
        }
        const single = line.match(/^\\[STAR\\] single task → (\\d+) task\\(s\\) dispatched concurrently$/);
        if (single) {
          transformed.push("### Routing");
          transformed.push("- split: `false`");
          transformed.push("- task_count: `" + single[1] + "`");
          continue;
        }
        const task = line.match(/^--- Task (\\d+): (.+) \\(via (.+)\\) ---$/);
        if (task) {
          transformed.push("");
          transformed.push("## Task " + task[1] + ": " + task[2]);
          transformed.push("_Model: `" + task[3] + "`_");
          transformed.push("");
          continue;
        }
        const done = line.match(/^\\[STAR\\] Done\\. (\\d+) task\\(s\\) completed\\.$/);
        if (done) {
          transformed.push("");
          transformed.push("---");
          transformed.push("**Done. " + done[1] + " task(s) completed.**");
          continue;
        }
        transformed.push(line);
      }
      return transformed.join("\\n");
    }

    function renderFinalOutput(rawText) {
      const markdown = toMarkdown(rawText);
      const html = marked.parse(markdown, {
        breaks: true,
        gfm: true,
      });
      outputRendered.innerHTML = html;
      outputRendered.querySelectorAll("pre code").forEach((block) => {
        hljs.highlightElement(block);
      });
      outputRaw.classList.add("hidden");
      outputRendered.classList.remove("hidden");
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
      outputRaw.textContent = "";
      outputRaw.classList.remove("hidden");
      outputRendered.classList.add("hidden");
      outputRendered.innerHTML = "";
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
        let fullText = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          fullText += chunk;
          outputRaw.textContent += chunk;
          outputRaw.scrollTop = outputRaw.scrollHeight;
          buffered += chunk;
          const lines = buffered.split("\\n");
          buffered = lines.pop() || "";
          for (const line of lines) parseLine(line, state);
          renderMeta(state);
        }
        if (buffered) parseLine(buffered, state);
        renderFinalOutput(fullText);
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
