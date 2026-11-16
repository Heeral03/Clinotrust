/* ClinoReview.mount(root, {turns, claims}, {filter}) -> {setFilter}. Used by the landing demo and the app. */
window.ClinoReview = (function () {
  const LABEL = { supported: "Supported", unsupported: "Unsupported", contradicted: "Contradicted", uncertain: "Uncertain" };
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  function highlight(node, text, quote) {
    const parts = (quote || "").split(/\s*\.\.\.\s*/).map(p => p.trim()).filter(Boolean);
    let ranges = [];
    for (const p of parts) {
      const m = new RegExp(p.split(/\s+/).map(esc).join("\\s*"), "i").exec(text);
      if (m) ranges.push([m.index, m.index + m[0].length]);
    }
    ranges.sort((a, b) => a[0] - b[0]);
    let pos = 0;
    for (const [s, e] of ranges) {
      if (s < pos) continue;
      node.append(text.slice(pos, s)); node.append(el("mark", "", text.slice(s, e))); pos = e;
    }
    node.append(text.slice(pos));
  }

  function mount(root, data, opts = {}) {
    let filter = opts.filter || "all", sel = null;
    root.textContent = "";
    root.className = "rv";
    const left = el("div", "rv-col"), right = el("div", "rv-col");
    const list = el("div", "rv-scroll"), why = el("div", "why"), tr = el("div", "rv-scroll");
    left.append(el("div", "label", "Note, split into claims"), list, why);
    right.append(el("div", "label", "Transcript"), tr);
    root.append(left, right);
    const visible = () => data.claims.filter(c => filter === "all" || (filter === "review" ? c.needs_review : c.status === filter));

    function renderTranscript() {
      tr.textContent = "";
      const c = data.claims.find(x => x.id === sel), ev = new Set(c ? c.evidence_turns : []);
      let first = null;
      for (const t of data.turns) {
        const d = el("div", "turn" + (ev.has(t.id) ? " ev" : ""));
        d.id = "turn-" + t.id;
        d.append(el("span", "tid", t.id), el("b", "", t.speaker[0].toUpperCase() + t.speaker.slice(1) + ": "));
        if (ev.has(t.id)) { highlight(d, t.text, c.quote); first = first || d; } else d.append(t.text);
        tr.append(d);
      }
      if (first) tr.scrollTo({ top: first.offsetTop - tr.offsetTop - 40, behavior: "smooth" });
    }
    function renderWhy() {
      why.textContent = "";
      const c = data.claims.find(x => x.id === sel);
      if (!c) { why.append(el("div", "label", "Why"), el("p", "", "Click a claim to see the evidence.")); return; }
      why.append(el("div", "label", "Why"), el("p", "", c.analysis || "(no analysis)"));
      why.append(el("div", "ev", c.evidence_turns.length ? "Evidence: " + c.evidence_turns.join(", ") : "Evidence: none found in the transcript"));
      if (c.quote_valid === false) why.append(el("div", "ev warnline", "Quote not found verbatim in the transcript; check manually."));
      if (c.reviewer) {
        const same = c.reviewer.status === c.status;
        why.append(el("div", "agree", (same ? "✓ Matches" : "✗ Differs from") + " the reviewer label: " + c.reviewer.status +
          (c.reviewer.error_type && c.reviewer.error_type !== "none" ? " (" + c.reviewer.error_type + ")" : "")));
      }
    }
    function select(id) { sel = id; renderList(); renderWhy(); renderTranscript(); }
    function renderList() {
      list.textContent = "";
      const vs = visible();
      if (!vs.length) list.append(el("div", "empty", "No claims match this filter."));
      for (const c of vs) {
        const b = el("button", "claim" + (c.id === sel ? " on" : ""));
        const txt = el("span", "");
        txt.append(el("span", "cid", c.id), c.text);
        if (c.section) txt.append(el("span", "sec", c.section));
        b.append(txt, el("span", "chip " + c.status, LABEL[c.status] || c.status));
        b.onclick = () => select(c.id);
        b.onkeydown = e => {
          const i = vs.findIndex(x => x.id === c.id), n = e.key === "ArrowDown" ? i + 1 : e.key === "ArrowUp" ? i - 1 : null;
          if (n != null && vs[n]) { e.preventDefault(); select(vs[n].id); list.querySelectorAll(".claim")[n].focus(); }
        };
        list.append(b);
      }
    }
    renderList(); renderWhy(); renderTranscript();
    return { setFilter(f) { filter = f; renderList(); }, select };
  }
  return { mount };
})();
