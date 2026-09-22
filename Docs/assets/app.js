/* Sandbox Docs — static site renderer. No build step, no external deps.
   Data lives in assets/content.js as window.DOCS_DATA. */

(function () {
  "use strict";

  const DATA = window.DOCS_DATA;
  const $sidebar = document.getElementById("sidebar-body");
  const $main = document.getElementById("main");
  const $tabs = document.getElementById("tabs");

  const TOP_TABS = [
    { id: "structure", label: "Project Structure", icon: "▦" },
    { id: "architecture", label: "Architecture", icon: "⚙" },
    { id: "pipelines", label: "Pipelines", icon: "⇄" },
    { id: "hld", label: "High-Level Design", icon: "◈" },
    { id: "files", label: "Files", icon: "☷" },
  ];

  /* ---------------- utilities ---------------- */

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function el(html) {
    const t = document.createElement("template");
    t.innerHTML = html.trim();
    return t.content.firstElementChild;
  }

  function tierBadge(tier) {
    const map = {
      tier0: "Tier 0",
      tier1: "Tier 1",
      tier2: "Tier 2",
      tier3: "Tier 3",
      config: "Config",
      data: "Data Fixture",
      test: "Test",
      doc: "Planning / Design Doc",
    };
    const cls = tier === "data" || tier === "doc" ? "doc" : tier;
    return `<span class="badge ${cls === "doc" ? "doc" : tier}">${map[tier] || tier}</span>`;
  }

  /* ---------------- tiny syntax highlighter ---------------- */
  // Deliberately small and dependency-free. Good enough for readable
  // Python / YAML / PlantUML / JSON / text code blocks in a docs site.

  function highlight(code, lang) {
    const src = escapeHtml(code);
    if (lang === "python") return highlightPython(src);
    if (lang === "yaml") return highlightYaml(src);
    if (lang === "plantuml") return highlightPlantuml(src);
    if (lang === "json") return highlightJson(src);
    return src;
  }

  const PY_KEYWORDS =
    /\b(def|class|return|if|elif|else|for|while|try|except|finally|raise|import|from|as|with|pass|continue|break|not|in|is|and|or|None|True|False|async|await|lambda|yield|global|nonlocal|assert|del|self)\b/g;

  function highlightPython(src) {
    const lines = src.split("\n").map((line) => {
      // comment (naive: from first unescaped # to end of line)
      const commentIdx = findCommentStart(line);
      let code = commentIdx === -1 ? line : line.slice(0, commentIdx);
      const comment = commentIdx === -1 ? "" : line.slice(commentIdx);

      code = code.replace(/(&quot;.*?&quot;|&#39;.*?&#39;)/g, '<span class="tok-str">$1</span>');
      code = code.replace(/^(\s*@\w[\w.]*)/, '<span class="tok-deco">$1</span>');
      code = code.replace(
        /\bdef\s+([A-Za-z_]\w*)/,
        'def <span class="tok-fn">$1</span>'
      );
      code = code.replace(
        /\bclass\s+([A-Za-z_]\w*)/,
        'class <span class="tok-cls">$1</span>'
      );
      code = code.replace(PY_KEYWORDS, '<span class="tok-kw">$1</span>');
      code = code.replace(/\b(\d+\.?\d*)\b/g, '<span class="tok-num">$1</span>');

      return comment
        ? code + '<span class="tok-com">' + comment + "</span>"
        : code;
    });
    return lines.join("\n");
  }

  function findCommentStart(line) {
    let inStr = null;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (inStr) {
        if (c === inStr) inStr = null;
        continue;
      }
      if (c === '"' || c === "'") {
        inStr = c;
        continue;
      }
      if (c === "#") return i;
    }
    return -1;
  }

  function highlightYaml(src) {
    return src
      .split("\n")
      .map((line) => {
        const cIdx = line.indexOf("#");
        let code = cIdx === -1 ? line : line.slice(0, cIdx);
        const comment = cIdx === -1 ? "" : line.slice(cIdx);
        code = code.replace(
          /^(\s*(?:-\s*)?)([A-Za-z0-9_.\-]+)(\s*:)/,
          '$1<span class="tok-fn">$2</span>$3'
        );
        code = code.replace(/\b(\d+\.?\d*)\b/g, '<span class="tok-num">$1</span>');
        return comment ? code + '<span class="tok-com">' + comment + "</span>" : code;
      })
      .join("\n");
  }

  function highlightPlantuml(src) {
    return src
      .split("\n")
      .map((line) => {
        let code = line.replace(
          /\b(@startuml|@enduml|actor|participant|database|rectangle|package|component|interface|note|left|right|over|of|as|alt|else|end|loop|opt|par|activate|deactivate|autonumber|title)\b/g,
          '<span class="tok-kw">$1</span>'
        );
        code = code.replace(/(&quot;.*?&quot;)/g, '<span class="tok-str">$1</span>');
        return code;
      })
      .join("\n");
  }

  function highlightJson(src) {
    return src.replace(
      /(&quot;.*?&quot;)(\s*:)?/g,
      (m, str, colon) =>
        colon ? `<span class="tok-fn">${str}</span>${colon}` : `<span class="tok-str">${str}</span>`
    );
  }

  function codeBlock(code, lang) {
    return `<pre><code>${highlight(code, lang || "text")}</code></pre>`;
  }

  /* ---------------- sidebar ---------------- */

  function renderSidebar(activeTab, activeFileKey) {
    let html = `
      <div class="search-box">
        <span class="icon">⌕</span>
        <input id="search-input" type="text" placeholder="Search files, functions, topics..." autocomplete="off" />
      </div>
      <div class="nav-section">
    `;
    for (const t of TOP_TABS) {
      if (t.id === "files") continue;
      html += `<div class="nav-link ${activeTab === t.id ? "active" : ""}" data-nav="#/${t.id}">
        <span class="ic">${t.icon}</span>${t.label}
      </div>`;
    }
    html += `</div><div class="files-heading">Files</div><div id="file-tree"></div>`;
    $sidebar.innerHTML = html;

    const treeRoot = document.getElementById("file-tree");
    treeRoot.appendChild(renderTree(DATA.fileTree, activeFileKey, ""));

    document.getElementById("search-input").addEventListener("input", (e) => {
      applySearchFilter(e.target.value.trim().toLowerCase());
    });

    $sidebar.querySelectorAll("[data-nav]").forEach((n) => {
      n.addEventListener("click", () => {
        location.hash = n.getAttribute("data-nav");
      });
    });
  }

  function renderTree(nodes, activeFileKey, pathPrefix) {
    const ul = document.createElement("ul");
    ul.className = "tree";
    ul.style.paddingLeft = pathPrefix ? "14px" : "0";
    for (const node of nodes) {
      const li = document.createElement("li");
      if (node.type === "folder") {
        const folderId = "f_" + (pathPrefix + node.name).replace(/[^a-z0-9]/gi, "_");
        const head = el(`<div class="tree-folder" data-folder="${folderId}">
          <span class="chev">▾</span><span class="fname">${escapeHtml(node.name)}</span>
        </div>`);
        li.appendChild(head);
        const childWrap = renderTree(node.children, activeFileKey, pathPrefix + node.name + "/");
        li.appendChild(childWrap);
        head.addEventListener("click", () => {
          head.classList.toggle("collapsed");
          childWrap.classList.toggle("hidden-by-search-manual");
          childWrap.style.display = childWrap.style.display === "none" ? "" : "none";
        });
      } else {
        const fileDoc = DATA.files[node.key];
        const tierClass = fileDoc ? fileDoc.tier : "doc";
        const dotColor = "var(--" + (tierClass === "data" || tierClass === "doc" ? "text-faint" : tierClass) + ")";
        const a = el(`<a class="tree-file ${node.key === activeFileKey ? "active" : ""}" data-file="${node.key}" data-search="${escapeHtml((node.name + " " + node.key).toLowerCase())}">
          <span class="dot" style="background:${dotColor}"></span>${escapeHtml(node.name)}
        </a>`);
        a.addEventListener("click", () => {
          location.hash = "#/files/" + encodeURIComponent(node.key);
        });
        li.appendChild(a);
      }
      ul.appendChild(li);
    }
    return ul;
  }

  function applySearchFilter(q) {
    const tree = document.getElementById("file-tree");
    if (!q) {
      tree.querySelectorAll("li").forEach((li) => (li.style.display = ""));
      tree.querySelectorAll(".tree-file").forEach((f) => f.classList.remove("searchable-hit"));
      return;
    }
    tree.querySelectorAll(".tree > li").forEach((li) => walkFilter(li, q));
  }

  function walkFilter(li, q) {
    const fileEl = li.querySelector(":scope > .tree-file");
    if (fileEl) {
      const hay = fileEl.getAttribute("data-search") || "";
      const match = hay.includes(q);
      li.style.display = match ? "" : "none";
      fileEl.classList.toggle("searchable-hit", match);
      return match;
    }
    const childLis = li.querySelectorAll(":scope > ul > li");
    let anyMatch = false;
    childLis.forEach((child) => {
      if (walkFilter(child, q)) anyMatch = true;
    });
    li.style.display = anyMatch ? "" : "none";
    if (anyMatch) {
      const folderHead = li.querySelector(":scope > .tree-folder");
      if (folderHead) {
        folderHead.classList.remove("collapsed");
        const childUl = li.querySelector(":scope > ul");
        if (childUl) childUl.style.display = "";
      }
    }
    return anyMatch;
  }

  /* ---------------- page renderers ---------------- */

  function pageHeader(title, subtitle) {
    return `<div class="page"><h1>${escapeHtml(title)}</h1>${
      subtitle ? `<div class="subtitle">${subtitle}</div>` : ""
    }`;
  }

  function renderStructure() {
    const d = DATA.structure;
    let html = pageHeader("Project Structure", "What lives where in this repository, and why.");
    html += `<p>${d.intro}</p>`;
    html += `<div class="tree-diagram">${escapeHtml(d.tree)}</div>`;
    html += `<h2>Legend</h2><div class="legend">`;
    for (const item of d.legend) {
      html += `<div class="legend-item"><span class="legend-dot" style="background:var(--${item.color})"></span>${item.label}</div>`;
    }
    html += `</div>`;
    html += `<h2>Folder-by-folder</h2>`;
    for (const f of d.folders) {
      html += `<div class="folder-entry"><div class="fp">${escapeHtml(f.path)}</div><p>${f.description}</p></div>`;
    }
    html += `</div>`;
    $main.innerHTML = html;
  }

  function renderArchitecture() {
    const d = DATA.architecture;
    let html = pageHeader("Architecture", "The major functional components of the sandbox and how they relate.");
    html += `<p>${d.intro}</p>`;
    html += `<div class="arch-nav">`;
    for (const c of d.components) {
      html += `<span class="pill-link" data-jump="${c.id}">${escapeHtml(c.name)}</span>`;
    }
    html += `</div>`;

    for (const c of d.components) {
      html += `<h2 id="arch-${c.id}">${escapeHtml(c.name)} <span style="font-weight:400;font-size:12.5px;color:var(--text-faint)">${c.tierLabel || ""}</span></h2>`;
      html += `<p><strong>What it is:</strong> ${c.whatItIs}</p>`;
      html += `<p><strong>What it does:</strong> ${c.whatItDoes}</p>`;
      if (c.files && c.files.length) {
        html += `<div class="io-block"><div class="io-label">Implemented by</div><div class="comm-list">`;
        for (const f of c.files) {
          html += DATA.files[f]
            ? `<span class="comm-tag" style="cursor:pointer" data-filejump="${f}">${escapeHtml(f)}</span>`
            : `<span class="comm-tag">${escapeHtml(f)}</span>`;
        }
        html += `</div></div>`;
      }
      if (c.communicatesWith && c.communicatesWith.length) {
        html += `<div class="io-block"><div class="io-label">Communicates with</div><div class="comm-list">`;
        for (const w of c.communicatesWith) html += `<span class="comm-tag">${escapeHtml(w)}</span>`;
        html += `</div></div>`;
      }
      if (c.dataFlow) {
        html += `<div class="io-block"><div class="io-label">Data flow</div><p style="margin-top:2px">${c.dataFlow}</p></div>`;
      }
    }
    html += `</div>`;
    $main.innerHTML = html;

    $main.querySelectorAll("[data-jump]").forEach((n) => {
      n.addEventListener("click", () => {
        document.getElementById("arch-" + n.getAttribute("data-jump")).scrollIntoView({ behavior: "smooth" });
      });
    });
    $main.querySelectorAll("[data-filejump]").forEach((n) => {
      n.addEventListener("click", () => {
        location.hash = "#/files/" + encodeURIComponent(n.getAttribute("data-filejump"));
      });
    });
  }

  function renderPipelines() {
    let html = pageHeader("Pipelines", "The end-to-end execution and data flows traced directly from the code and the design documents.");
    html += `<div class="pipeline-nav">`;
    for (const p of DATA.pipelines) {
      html += `<span class="pill-link" data-jump="p-${p.id}">${escapeHtml(p.name)}</span>`;
    }
    html += `</div>`;
    for (const p of DATA.pipelines) {
      html += `<h2 id="p-${p.id}">${escapeHtml(p.name)}</h2>`;
      if (p.status) html += `<p>${tierBadge(p.status)}</p>`;
      html += `<p>${p.description}</p>`;
      html += `<h3>Sequence diagram (PlantUML source)</h3>`;
      html += codeBlock(p.plantuml, "plantuml");
    }
    html += `</div>`;
    $main.innerHTML = html;
  }

  function renderHLD() {
    const d = DATA.hld;
    let html = pageHeader("High-Level Design", "System-level view of the sandbox: components, boundaries, and data flow.");
    for (const section of d.sections) {
      html += `<h2>${escapeHtml(section.title)}</h2>`;
      for (const p of section.paragraphs) html += `<p>${p}</p>`;
      if (section.list) {
        html += "<ul>";
        for (const item of section.list) html += `<li>${item}</li>`;
        html += "</ul>";
      }
    }
    html += `<h2>PlantUML source</h2>`;
    html += `<p>Rendered separately by the maintainer; shown here as source only.</p>`;
    html += codeBlock(d.plantuml, "plantuml");
    html += `</div>`;
    $main.innerHTML = html;
  }

  function renderFilesIndex() {
    let html = pageHeader("Files", "Every source, configuration, data-fixture, test, and planning file documented in this repository. Pick a file from the sidebar, or browse by tier below.");
    const groups = DATA.filesIndexGroups;
    for (const g of groups) {
      html += `<h2>${escapeHtml(g.title)}</h2><p>${g.description || ""}</p><div class="card-grid">`;
      for (const key of g.keys) {
        const f = DATA.files[key];
        if (!f) continue;
        html += `<div class="card" style="cursor:pointer" data-filejump="${key}">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px">
            <span class="path-pill">${escapeHtml(f.path)}</span>${tierBadge(f.tier)}
          </div>
          <p style="margin:10px 0 0;font-size:13.5px">${f.oneLiner}</p>
        </div>`;
      }
      html += `</div>`;
    }
    html += `</div>`;
    $main.innerHTML = html;
    $main.querySelectorAll("[data-filejump]").forEach((n) => {
      n.addEventListener("click", () => {
        location.hash = "#/files/" + encodeURIComponent(n.getAttribute("data-filejump"));
      });
    });
  }

  function renderParam(p) {
    return `<div class="param-row">
      <span class="pname">${escapeHtml(p.name)}</span>
      ${p.type ? `<span class="ptype">: ${escapeHtml(p.type)}</span>` : ""}
      <div style="color:var(--text-dim);font-size:13px;margin-top:1px">${p.desc}</div>
    </div>`;
  }

  function renderMember(m) {
    let html = `<div class="member-card">
      <div class="member-head">
        <span class="kind-tag">${m.kind}</span>
        <span>${escapeHtml(m.signature)}</span>
      </div>
      <div class="member-body">`;
    if (m.description) html += `<p>${m.description}</p>`;
    if (m.input && m.input.length) {
      html += `<div class="io-block"><div class="io-label">Input</div>`;
      for (const p of m.input) html += renderParam(p);
      html += `</div>`;
    } else if (m.input) {
      html += `<div class="io-block"><div class="io-label">Input</div><div class="param-row" style="color:var(--text-dim)">None</div></div>`;
    }
    if (m.output) {
      html += `<div class="io-block"><div class="io-label">Output</div>
        <div class="param-row"><span class="ptype">${escapeHtml(m.output.type || "")}</span>
        <div style="color:var(--text-dim);font-size:13px;margin-top:1px">${m.output.desc}</div></div></div>`;
    }
    if (m.behavior) {
      html += `<div class="io-block"><div class="io-label">What it does</div><p style="margin:2px 0 0">${m.behavior}</p></div>`;
    }
    if (m.notes) {
      html += `<div class="callout">${m.notes}</div>`;
    }
    html += `</div></div>`;
    return html;
  }

  function renderFileDoc(key) {
    const f = DATA.files[key];
    if (!f) {
      $main.innerHTML = `<div class="page"><div class="empty-state">File not found: ${escapeHtml(key)}</div></div>`;
      return;
    }
    let html = `<div class="page">`;
    html += `<div class="crumbs"><a href="#/files">Files</a> / ${escapeHtml(f.path)}</div>`;
    html += `<h1>${escapeHtml(f.path.split("/").pop())}</h1>`;
    html += `<div class="subtitle"><span class="path-pill">${escapeHtml(f.path)}</span> &nbsp; ${tierBadge(f.tier)}</div>`;

    html += `<h2>Overview</h2>`;
    for (const p of f.overview) html += `<p>${p}</p>`;

    if (f.dependencies && f.dependencies.length) {
      html += `<div class="io-block"><div class="io-label">Depends on</div><div class="comm-list">`;
      for (const dep of f.dependencies) {
        html += DATA.files[dep]
          ? `<span class="comm-tag" style="cursor:pointer" data-filejump="${dep}">${escapeHtml(dep)}</span>`
          : `<span class="comm-tag">${escapeHtml(dep)}</span>`;
      }
      html += `</div></div>`;
    }
    if (f.dependents && f.dependents.length) {
      html += `<div class="io-block"><div class="io-label">Used by</div><div class="comm-list">`;
      for (const dep of f.dependents) {
        html += DATA.files[dep]
          ? `<span class="comm-tag" style="cursor:pointer" data-filejump="${dep}">${escapeHtml(dep)}</span>`
          : `<span class="comm-tag">${escapeHtml(dep)}</span>`;
      }
      html += `</div></div>`;
    }

    if (f.classes && f.classes.length) {
      html += `<h2>Classes</h2>`;
      for (const c of f.classes) {
        html += `<div class="member-card"><div class="member-head"><span class="kind-tag">class</span><span class="tok-cls" style="font-family:ui-monospace,monospace">${escapeHtml(
          c.name
        )}</span></div><div class="member-body">`;
        if (c.purpose) html += `<p>${c.purpose}</p>`;
        if (c.attributes && c.attributes.length) {
          html += `<div class="io-block"><div class="io-label">Important attributes</div>`;
          for (const a of c.attributes) html += renderParam(a);
          html += `</div>`;
        }
        if (c.constructor_) {
          html += `<div class="io-block"><div class="io-label">Constructor</div><p class="mono" style="font-size:12.8px;background:var(--bg-code);border:1px solid var(--border);border-radius:6px;padding:8px 10px">${escapeHtml(
            c.constructor_.signature
          )}</p>`;
          for (const a of c.constructor_.input || []) html += renderParam(a);
          html += `</div>`;
        }
        html += `</div></div>`;
        if (c.methods && c.methods.length) {
          for (const m of c.methods) html += renderMember(Object.assign({ kind: "method" }, m));
        }
      }
    }

    if (f.functions && f.functions.length) {
      html += `<h2>Functions</h2>`;
      for (const fn of f.functions) html += renderMember(Object.assign({ kind: "function" }, fn));
    }

    if (f.testCoverage && f.testCoverage.length) {
      html += `<h2>What this test file verifies</h2><ul>`;
      for (const t of f.testCoverage) html += `<li>${t}</li>`;
      html += `</ul>`;
    }

    if (f.notes && f.notes.length) {
      html += `<h2>Notes</h2>`;
      for (const n of f.notes) html += `<div class="callout">${n}</div>`;
    }

    html += renderPrevNext(key);
    html += `</div>`;
    $main.innerHTML = html;

    $main.querySelectorAll("[data-filejump]").forEach((n) => {
      n.addEventListener("click", () => {
        location.hash = "#/files/" + encodeURIComponent(n.getAttribute("data-filejump"));
      });
    });
  }

  function flatFileKeys() {
    const out = [];
    (function walk(nodes) {
      for (const n of nodes) {
        if (n.type === "file") out.push(n.key);
        else walk(n.children);
      }
    })(DATA.fileTree);
    return out;
  }

  function renderPrevNext(key) {
    const keys = flatFileKeys();
    const idx = keys.indexOf(key);
    if (idx === -1) return "";
    const prev = keys[idx - 1];
    const next = keys[idx + 1];
    let html = `<div class="prevnext">`;
    html += prev
      ? `<a data-filejump="${prev}"><span class="lbl">← Previous</span>${escapeHtml(DATA.files[prev].path)}</a>`
      : `<span></span>`;
    html += next
      ? `<a data-filejump="${next}" style="text-align:right"><span class="lbl">Next →</span>${escapeHtml(DATA.files[next].path)}</a>`
      : `<span></span>`;
    html += `</div>`;
    return html;
  }

  /* ---------------- router ---------------- */

  function renderTabs(activeTab) {
    $tabs.innerHTML = TOP_TABS.map(
      (t) =>
        `<button class="tab-btn ${activeTab === t.id ? "active" : ""}" data-tab="${t.id}">${t.label}</button>`
    ).join("");
    $tabs.querySelectorAll("[data-tab]").forEach((b) => {
      b.addEventListener("click", () => {
        const id = b.getAttribute("data-tab");
        location.hash = id === "files" ? "#/files" : "#/" + id;
      });
    });
  }

  function route() {
    const hash = location.hash || "#/structure";
    const parts = hash.replace(/^#\//, "").split("/");
    const tab = parts[0] || "structure";

    let activeFileKey = null;
    if (tab === "files" && parts[1]) {
      activeFileKey = decodeURIComponent(parts.slice(1).join("/"));
    }

    renderTabs(tab === "files" ? "files" : tab);
    renderSidebar(tab, activeFileKey);

    if (tab === "structure") renderStructure();
    else if (tab === "architecture") renderArchitecture();
    else if (tab === "pipelines") renderPipelines();
    else if (tab === "hld") renderHLD();
    else if (tab === "files") {
      if (activeFileKey) renderFileDoc(activeFileKey);
      else renderFilesIndex();
    } else renderStructure();

    $main.scrollTop = 0;
  }

  window.addEventListener("hashchange", route);

  /* ---------------- theme toggle ---------------- */

  function initTheme() {
    const saved = localStorage.getItem("sandbox-docs-theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
    const btn = document.getElementById("theme-toggle");
    btn.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme");
      const next = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("sandbox-docs-theme", next);
      btn.textContent = next === "dark" ? "☀" : "☽";
    });
    btn.textContent = document.documentElement.getAttribute("data-theme") === "dark" ? "☀" : "☽";
  }

  initTheme();
  route();
})();
