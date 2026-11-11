/*
  The page itself: draws the pitch, plays the clips, lets you drag
  players, and asks the engine for ghosts every time something changes.
*/
(function () {
  "use strict";
  const E = window.GhostEngine;
  const DATA = window.GHOST_DATA;
  const $ = (id) => document.getElementById(id);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const COLOR = { turf: css("--turf"), turf2: css("--turf-2"), chalk: css("--chalk"), ghost: css("--ghost"),
                  attack: css("--attack"), defend: css("--defend"), alarm: css("--alarm"), night: css("--night") };

  const model = new E.Model(E.decodeWeights(DATA.weights.data, DATA.weights.manifest));

  // ---------- clips ----------

  const LOOK_BACK = 8; // ghosts start from where the defenders stood 8 seconds before the shot
  const clips = DATA.clips.map((raw) => {
    const clip = E.unpackClip(raw);
    clip.start = Math.max(0, clip.shot - Math.round(LOOK_BACK * clip.fps));
    clip.cache = new Map();
    clip.label = clip.name.replace(/[\[\]]/g, "");
    // A goal that starts with the ball at a corner flag is a corner kick.
    clip.corner = Math.abs(clip.ball[0]) > 50 && Math.abs(clip.ball[1]) > 31.5;
    let left = Infinity;
    for (const list of [clip.ball, clip.att, clip.def]) for (let i = 0; i < list.length; i += 2) left = Math.min(left, list[i]);
    clip.view = { x0: Math.min(-22, Math.max(-54.5, left - 5)), x1: 56, y0: -36, y1: 36 };
    return clip;
  });

  const state = { clip: clips[0], frame: 0, playing: false, mode: "team", map: "off", moved: null, highlight: -1 };

  // Everything we show for one frame: ghosts, danger numbers, distances.
  function teamGhostAt(clip, frame) {
    if (frame <= clip.start) return E.momentOf(clip, frame).def;
    const hit = clip.cache.get(frame);
    if (hit) return hit.team;
    return E.teamGhost(model, E.momentOf(clip, frame), E.momentOf(clip, clip.start), (frame - clip.start) / clip.fps);
  }

  function work(clip, frame, moved) {
    if (!moved && clip.cache.has(frame)) return clip.cache.get(frame);
    const now = moved || E.momentOf(clip, frame);
    const out = { now, released: frame > clip.start };
    if (out.released) {
      const then = E.momentOf(clip, clip.start);
      const seconds = (frame - clip.start) / clip.fps;
      out.team = E.teamGhost(model, now, then, seconds);
      out.solo = E.soloGhosts(model, now, then, seconds);
      const before = Math.max(frame - clip.halfSecond, 0);
      out.ghostV = E.ghostSpeed(moved ? teamGhostAt(clip, frame) : out.team, teamGhostAt(clip, before), (frame - before) / clip.fps);
    } else {
      out.team = now.def; out.solo = now.def; out.ghostV = now.defV;
    }
    out.dangerReal = E.danger(now.att, now.attV, now.def, now.defV);
    out.dangerGhost = E.danger(now.att, now.attV, out.team, out.ghostV);
    out.offBy = [];
    for (let i = 0; i < clip.nDef; i++) out.offBy.push(Math.hypot(now.def[i * 2] - out.solo[i * 2], now.def[i * 2 + 1] - out.solo[i * 2 + 1]));
    if (!moved) clip.cache.set(frame, out);
    return out;
  }

  // Fill the cache in the background so playback is smooth.
  function warmUp(clip) {
    let frame = 0;
    (function next() {
      if (state.clip !== clip || frame >= clip.frames) return;
      work(clip, frame++, null);
      setTimeout(next, 0);
    })();
  }

  // ---------- drawing ----------

  const canvas = $("pitch");
  const ctx = canvas.getContext("2d");
  let scale = 1, width = 0, height = 0;

  function resize() {
    const view = state.clip.view;
    const box = canvas.parentElement.clientWidth - 20;
    const tallest = Math.max(300, window.innerHeight * 0.64);
    width = Math.min(box, tallest * (view.x1 - view.x0) / (view.y1 - view.y0));
    scale = width / (view.x1 - view.x0);
    height = (view.y1 - view.y0) * scale;
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    canvas.style.margin = "0 auto";
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  const px = (x) => (x - state.clip.view.x0) * scale;
  const py = (y) => (state.clip.view.y1 - y) * scale;

  function line(x0, y0, x1, y1) { ctx.beginPath(); ctx.moveTo(px(x0), py(y0)); ctx.lineTo(px(x1), py(y1)); ctx.stroke(); }
  function rect(x0, y0, x1, y1) { ctx.strokeRect(px(x0), py(y1), (x1 - x0) * scale, (y1 - y0) * scale); }
  function arc(x, y, r, from, to) { ctx.beginPath(); ctx.arc(px(x), py(y), r * scale, from, to); ctx.stroke(); }

  function drawPitch() {
    ctx.fillStyle = "#103c2c";
    ctx.fillRect(0, 0, width, height);
    // Mowing stripes.
    for (let i = 0; i < 20; i++) {
      ctx.fillStyle = i % 2 ? COLOR.turf : COLOR.turf2;
      const x0 = -52.5 + i * 5.25;
      ctx.fillRect(px(x0), py(34), 5.25 * scale + 0.5, 68 * scale);
    }
  }

  function drawLines() {
    ctx.strokeStyle = "rgba(241,245,238,.82)";
    ctx.lineWidth = Math.max(1, 0.18 * scale);
    rect(-52.5, -34, 52.5, 34);
    line(0, -34, 0, 34);
    arc(0, 0, 9.15, 0, Math.PI * 2);
    for (const s of [-1, 1]) {
      rect(Math.min(s * 52.5, s * 36), -20.16, Math.max(s * 52.5, s * 36), 20.16);
      rect(Math.min(s * 52.5, s * 47), -9.16, Math.max(s * 52.5, s * 47), 9.16);
      // The arc at the top of the box.
      const reach = Math.acos(5.5 / 9.15);
      if (s === 1) arc(41.5, 0, 9.15, Math.PI - reach, Math.PI + reach);
      else arc(-41.5, 0, 9.15, -reach, reach);
      ctx.fillStyle = "rgba(241,245,238,.82)";
      ctx.beginPath(); ctx.arc(px(s * 41.5), py(0), Math.max(1.5, 0.22 * scale), 0, Math.PI * 2); ctx.fill();
      // Goal.
      ctx.fillStyle = "rgba(241,245,238,.12)";
      ctx.fillRect(px(Math.min(s * 52.5, s * 54.6)), py(3.66), 2.1 * scale, 7.32 * scale);
      rect(Math.min(s * 52.5, s * 54.6), -3.66, Math.max(s * 52.5, s * 54.6), 3.66);
    }
  }

  // Shade the spots the attack would reach first.
  function drawMap(info) {
    const def = state.map === "ghost" ? info.team : info.now.def;
    const defV = state.map === "ghost" ? info.ghostV : info.now.defV;
    const control = E.attackControl(info.now.att, info.now.attV, def, defV);
    for (let s = 0; s < E.ZONE.count; s++) {
      const alpha = control[s] * (0.12 + 0.5 * Math.min(1, E.ZONE.weight[s] / 1.5));
      if (alpha < 0.02) continue;
      ctx.fillStyle = "rgba(255,178,36," + alpha.toFixed(3) + ")";
      ctx.fillRect(px(E.ZONE.spots[s * 2] - 0.5), py(E.ZONE.spots[s * 2 + 1] + 0.5), scale + 0.5, scale + 0.5);
    }
  }

  function dot(x, y, r, fill, stroke, strokeWidth) {
    ctx.beginPath(); ctx.arc(px(x), py(y), r, 0, Math.PI * 2);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = strokeWidth; ctx.stroke(); }
  }

  function draw() {
    const clip = state.clip;
    const info = work(clip, state.frame, state.moved);
    const now = info.now;
    const ghosts = state.mode === "team" ? info.team : info.solo;
    const r = Math.max(6, 1.05 * scale);

    ctx.clearRect(0, 0, width, height);
    drawPitch();
    if (state.map !== "off") drawMap(info);
    drawLines();

    if (info.released) {
      // A thread from each defender to his ghost. Amber when he is far away.
      for (let i = 0; i < clip.nDef; i++) {
        const gap = Math.hypot(now.def[i * 2] - ghosts[i * 2], now.def[i * 2 + 1] - ghosts[i * 2 + 1]);
        if (gap < 0.6) continue;
        ctx.strokeStyle = gap > 5 ? "rgba(255,178,36,.9)" : "rgba(214,255,240,.45)";
        ctx.lineWidth = i === state.highlight ? 2.5 : 1.25;
        ctx.setLineDash([4, 4]);
        line(now.def[i * 2], now.def[i * 2 + 1], ghosts[i * 2], ghosts[i * 2 + 1]);
        ctx.setLineDash([]);
      }
      // Ghosts: pale glowing rings.
      ctx.save();
      ctx.shadowColor = "rgba(214,255,240,.85)";
      ctx.shadowBlur = 12;
      for (let i = 0; i < clip.nDef; i++)
        dot(ghosts[i * 2], ghosts[i * 2 + 1], r, "rgba(214,255,240,.10)", COLOR.ghost, i === state.highlight ? 3.5 : 2);
      ctx.restore();
    }

    // Real defenders, with a number so the list on the right can point at them.
    ctx.font = "700 " + Math.max(9, r * 1.05) + "px 'Hanken Grotesk', sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (let i = 0; i < clip.nDef; i++) {
      dot(now.def[i * 2], now.def[i * 2 + 1], r, COLOR.defend, COLOR.chalk, i === state.highlight ? 3 : 1.5);
      ctx.fillStyle = COLOR.chalk;
      ctx.fillText(i === clip.defKeeper ? "K" : String(i + 1), px(now.def[i * 2]), py(now.def[i * 2 + 1]) + 0.5);
    }
    for (let i = 0; i < clip.nAtt; i++) dot(now.att[i * 2], now.att[i * 2 + 1], r, COLOR.attack, "rgba(8,19,15,.55)", 1.5);
    dot(now.ball[0], now.ball[1], Math.max(3.5, r * 0.5), "#fff", "#0b1a14", 1.5);

    updateReadouts(info);
  }

  // ---------- numbers beside the pitch ----------

  function updateReadouts(info) {
    const clip = state.clip;
    const toShot = (clip.shot - state.frame) / clip.fps;
    $("clock").innerHTML = Math.abs(toShot) < 0.5 / clip.fps ? "Shot" :
      Math.abs(toShot).toFixed(1) + " s<small>" + (toShot > 0 ? "to the shot" : "after the shot") + "</small>";

    const real = Math.round(info.dangerReal * 100), ghost = Math.round(info.dangerGhost * 100);
    $("barReal").style.width = real + "%"; $("numReal").textContent = real + "%";
    $("barGhost").style.width = ghost + "%"; $("numGhost").textContent = ghost + "%";
    $("dangerNote").textContent = !info.released ? "The ghosts have not started yet."
      : real - ghost >= 3 ? "The real defense leaves more open than the ghosts would."
      : ghost - real >= 3 ? "The real defense is tighter than the ghosts here."
      : "About the same.";

    // The three outfield defenders furthest from their own ghost.
    const order = info.offBy.map((v, i) => [v, i]).filter(([, i]) => i !== clip.defKeeper).sort((a, b) => b[0] - a[0]).slice(0, 3);
    const list = $("blame");
    while (list.children.length < order.length) {
      const li = document.createElement("li");
      li.innerHTML = '<span class="chip"></span><span class="who"></span><span class="num"></span>';
      li.addEventListener("pointerenter", () => { state.highlight = Number(li.dataset.index); draw(); });
      li.addEventListener("pointerleave", () => { state.highlight = -1; draw(); });
      list.appendChild(li);
    }
    order.forEach(([gap, i], n) => {
      const li = list.children[n];
      li.dataset.index = i;
      li.children[0].textContent = i + 1;
      li.children[1].textContent = "Defender " + (i + 1);
      li.children[2].textContent = info.released ? gap.toFixed(1) + " m" : "0.0 m";
      li.children[2].className = "num" + (gap > 5 && info.released ? " far" : "");
    });

    $("reset").hidden = !state.moved;
    $("badge").innerHTML = state.moved ? "You moved things. <b>The ghosts are reacting live.</b>" : "Attack plays left to right. <b>Drag the ball or a player.</b>";
  }

  // ---------- controls ----------

  function setFrame(frame) {
    state.frame = Math.max(0, Math.min(state.clip.frames - 1, Math.round(frame)));
    state.moved = null;
    $("scrub").value = state.frame;
    draw();
  }

  function setClip(index) {
    stop();
    state.clip = clips[(index + clips.length) % clips.length];
    state.highlight = -1;
    const clip = state.clip;
    $("goal").value = clips.indexOf(clip);
    $("scrub").max = clip.frames - 1;
    $("tickRelease").style.left = (clip.start / (clip.frames - 1) * 100) + "%";
    $("tickShot").style.left = (clip.shot / (clip.frames - 1) * 100) + "%";
    $("clipNote").textContent = "Tracked from TV footage. " + clip.nDef + " defenders and " + clip.nAtt + " attackers are in this clip." +
      (clip.corner ? " This goal starts from a corner, so trust these ghosts less." : "");
    $("clipNote").className = "note" + (clip.corner ? " warn" : "");
    resize();
    setFrame(clip.shot);
    warmUp(clip);
  }

  let lastTick = 0, carry = 0;
  function tick(time) {
    if (!state.playing) return;
    carry += (time - lastTick) / 1000 * state.clip.fps;
    lastTick = time;
    const steps = Math.floor(carry);
    if (steps > 0) {
      carry -= steps;
      if (state.frame + steps >= state.clip.frames - 1) { setFrame(state.clip.frames - 1); stop(); return; }
      setFrame(state.frame + steps);
    }
    requestAnimationFrame(tick);
  }
  function play() {
    if (state.frame >= state.clip.frames - 1) setFrame(0);
    state.playing = true; carry = 0; lastTick = performance.now();
    $("playIcon").innerHTML = '<path d="M3 1.5h3.5v13H3zM9.5 1.5H13v13H9.5z"/>';
    $("play").setAttribute("aria-label", "Pause");
    requestAnimationFrame(tick);
  }
  function stop() {
    state.playing = false;
    $("playIcon").innerHTML = '<path d="M3 1.5v13l11-6.5z"/>';
    $("play").setAttribute("aria-label", "Play");
  }

  $("play").addEventListener("click", () => (state.playing ? stop() : play()));
  $("scrub").addEventListener("input", (e) => { stop(); setFrame(Number(e.target.value)); });
  $("prev").addEventListener("click", () => setClip(clips.indexOf(state.clip) - 1));
  $("next").addEventListener("click", () => setClip(clips.indexOf(state.clip) + 1));
  $("goal").addEventListener("change", (e) => setClip(Number(e.target.value)));
  $("reset").addEventListener("click", () => { state.moved = null; draw(); });

  const MODE_NOTE = {
    team: "Whole unit: the model sees only the ball and the attackers, then places every defender.",
    solo: "One by one: each ghost is placed with his real teammates left where they were.",
  };
  function pressOne(group, attribute, value) {
    for (const button of group.querySelectorAll("button")) button.setAttribute("aria-pressed", String(button.dataset[attribute] === value));
  }
  $("modeSeg").addEventListener("click", (e) => {
    const mode = e.target.dataset.mode; if (!mode) return;
    state.mode = mode; pressOne($("modeSeg"), "mode", mode); $("modeNote").textContent = MODE_NOTE[mode]; draw();
  });
  $("mapSeg").addEventListener("click", (e) => {
    const map = e.target.dataset.map; if (!map) return;
    state.map = map; pressOne($("mapSeg"), "map", map); draw();
  });
  $("modeNote").textContent = MODE_NOTE.team;

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT" || e.target.tagName === "BUTTON") return;
    if (e.code === "Space") { e.preventDefault(); state.playing ? stop() : play(); }
    if (e.code === "ArrowRight") { stop(); setFrame(state.frame + 1); }
    if (e.code === "ArrowLeft") { stop(); setFrame(state.frame - 1); }
  });

  // ---------- dragging ----------

  let grabbed = null;
  function pitchPoint(event) {
    const box = canvas.getBoundingClientRect();
    return [state.clip.view.x0 + (event.clientX - box.left) / scale, state.clip.view.y1 - (event.clientY - box.top) / scale];
  }
  canvas.addEventListener("pointerdown", (event) => {
    const [x, y] = pitchPoint(event);
    const now = state.moved || E.momentOf(state.clip, state.frame);
    let best = null, bestGap = Math.max(2.2, 20 / scale);
    const consider = (list, i, kind) => { const gap = Math.hypot(list[i * 2] - x, list[i * 2 + 1] - y); if (gap < bestGap) { bestGap = gap; best = { kind, i }; } };
    consider(now.ball, 0, "ball");
    for (let i = 0; i < state.clip.nAtt; i++) consider(now.att, i, "att");
    for (let i = 0; i < state.clip.nDef; i++) consider(now.def, i, "def");
    if (!best) return;
    stop();
    state.moved = now;
    grabbed = best;
    canvas.classList.add("dragging");
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!grabbed) return;
    const [x, y] = pitchPoint(event);
    const list = state.moved[grabbed.kind];
    list[grabbed.i * 2] = Math.max(-54, Math.min(54, x));
    list[grabbed.i * 2 + 1] = Math.max(-35.5, Math.min(35.5, y));
    draw();
  });
  const release = () => { grabbed = null; canvas.classList.remove("dragging"); };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);

  // ---------- results section ----------

  function fillResults() {
    const R = DATA.results || {};
    if (R.model) {
      const avg = R.model.average, secs = Object.keys(avg).map(Number).sort((a, b) => a - b);
      const lines = [
        { key: "stand_still", name: "Stand still", color: "#6f8a7e", dash: "2 5" },
        { key: "follow_ball", name: "Move with the ball", color: "#9bb1a6", dash: "7 5" },
        { key: "model", name: "Ghost XI", color: COLOR.ghost, dash: "" },
      ];
      const top = Math.ceil(Math.max(...secs.map((s) => avg[s].all_hidden.stand_still)) / 2) * 2;
      const X = (s) => 46 + (s - secs[0]) / (secs[secs.length - 1] - secs[0]) * 380, Y = (m) => 286 - m / top * 262;
      let svg = "";
      for (let m = 0; m <= top; m += 2) svg += `<line x1="46" x2="426" y1="${Y(m)}" y2="${Y(m)}" stroke="#1d362d"/><text x="38" y="${Y(m) + 4}" text-anchor="end" font-size="12" fill="#9bb1a6">${m}</text>`;
      for (const s of secs) svg += `<text x="${X(s)}" y="308" text-anchor="middle" font-size="12" fill="#9bb1a6">${s} s</text>`;
      svg += `<text x="236" y="328" text-anchor="middle" font-size="12" fill="#9bb1a6">seconds ahead</text><text x="46" y="14" font-size="12" fill="#9bb1a6">meters</text>`;
      for (const l of lines) {
        const pts = secs.map((s) => [X(s), Y(avg[s].all_hidden[l.key])]);
        svg += `<polyline fill="none" stroke="${l.color}" stroke-width="${l.key === "model" ? 3 : 2}" stroke-dasharray="${l.dash}" points="${pts.map((p) => p.join(",")).join(" ")}"/>`;
        for (const p of pts) svg += `<circle cx="${p[0]}" cy="${p[1]}" r="${l.key === "model" ? 4 : 3}" fill="${l.color}"/>`;
        const end = pts[pts.length - 1], value = avg[secs[secs.length - 1]].all_hidden[l.key];
        svg += `<text x="${end[0] + 10}" y="${end[1] - 2}" font-size="13" font-weight="${l.key === "model" ? 700 : 400}" fill="${l.color}">${l.name}</text><text x="${end[0] + 10}" y="${end[1] + 13}" font-size="12" fill="${l.color}">${value.toFixed(1)} m</text>`;
      }
      $("chart").innerHTML = svg;

      const five = avg["5"];
      const facts = [];
      facts.push(`<article><div class="big"><em>${five.all_hidden.model.toFixed(1)} m</em></div><p>Average miss when guessing 5 seconds ahead. The best simple rule, <b>move with the ball</b>, misses by ${five.all_hidden.follow_ball.toFixed(1)} m.</p></article>`);
      facts.push(`<article><div class="big">${five.one_hidden.model.toFixed(1)} m <span style="font-size:.5em;color:var(--mist)">vs</span> ${five.one_hidden.copy_teammates.toFixed(1)} m</div><p>One defender hidden, 5 seconds ahead. Ghost XI against the rule <b>move like your teammates</b>.</p></article>`);
      const tv = R.model.by_source.tv_tracking["5"].all_hidden, stadium = R.model.by_source.stadium_tracking["5"].all_hidden;
      facts.push(`<article><div class="big">${tv.model.toFixed(1)} m <span style="font-size:.5em;color:var(--mist)">and</span> ${stadium.model.toFixed(1)} m</div><p>The same 5 second test on the <b>${R.model.tv_matches} matches tracked from TV</b>, where part of the pitch is off screen, and on the ${R.model.stadium_matches} tracked in the stadium.</p></article>`);
      if (R.camera) facts.push(`<article><div class="big">${R.camera.position_error_m.typical.toFixed(2)} m</div><p>Typical position error after going <b>pitch to pixels and back</b> through a pretend TV camera, with ${R.camera.players_in_picture_out_of_22} of 22 players in the picture on average.</p></article>`);
      facts.push(`<article><div class="big">${R.model.parameters.toLocaleString("en-US")}</div><p>Numbers inside the model. Small enough to run in this page. <b id="speedFact"></b></p></article>`);
      $("facts").innerHTML = facts.join("");
      $("trainSteps").textContent = (R.model.steps * 256).toLocaleString("en-US");
      if (R.model.moments) $("trainMoments").textContent = R.model.moments.toLocaleString("en-US");
    }
    if (R.goals) {
      const g = R.goals.groups;
      const rows = [["ordinary", "Ordinary moments near goal"], ["shot", "Shots that did not go in"], ["goal", "Goals"], ["tv_goal", "Liverpool goals (other data source)"]];
      let html = "<thead><tr><th>At this kind of moment</th><th>How many</th><th>Average defender is off by</th><th>Worst defender is off by</th><th>Space won by attack, real</th><th>Space won by attack, ghosts</th></tr></thead><tbody>";
      for (const [key, name] of rows) {
        const x = g[key]; if (!x || !x.count) continue;
        html += `<tr><td>${name}</td><td>${x.count}</td><td>${x.off_by_m.mean.toFixed(1)} m</td><td class="hi">${x.worst_off_by_m.mean.toFixed(1)} m</td><td>${Math.round(x.danger_real.mean * 100)}%</td><td>${Math.round(x.danger_ghost.mean * 100)}%</td></tr>`;
      }
      html += "</tbody><caption>Ghosts start 8 seconds before each moment. The first three rows are from matches the model did not train on. The Liverpool goals come from a different data source, so they are new to the model as well. The space numbers are not comparable between rows: TV tracking shows fewer players, and fewer players on screen changes the number.</caption>";
      $("groups").innerHTML = html;
      // Say plainly what the table shows, using the numbers themselves.
      if (g.goal && g.goal.count && g.ordinary && g.ordinary.count) {
        const goalGap = g.goal.off_by_m, plainGap = g.ordinary.off_by_m;
        const apart = goalGap.low > plainGap.high || goalGap.high < plainGap.low;
        $("verdict").innerHTML = apart
          ? `<b>What this shows.</b> Before a goal, defenders are ${goalGap.mean.toFixed(1)} m from their ghosts on average. At an ordinary moment it is ${plainGap.mean.toFixed(1)} m. That gap is bigger than chance would explain.`
          : `<b>What this shows, honestly.</b> Before a goal, defenders are ${goalGap.mean.toFixed(1)} m from their ghosts on average. At an ordinary moment it is ${plainGap.mean.toFixed(1)} m. That is too close to call a difference. The ghost defense also gives up about as much space as the real one. So a defender far from his ghost did something <b>unusual</b>. On its own that does not mean he made a <b>mistake</b>.`;
      }
    }
  }

  // How long does one run of the model take on this device?
  function timeModel() {
    const clip = clips[0], then = E.momentOf(clip, clip.start), now = E.momentOf(clip, clip.shot);
    const began = performance.now();
    for (let i = 0; i < 30; i++) E.teamGhost(model, now, then, 5);
    const ms = (performance.now() - began) / 30;
    if (!(ms > 0.05)) return;   // some browsers blur their clock. Say nothing if we could not measure.
    if ($("speedFact")) $("speedFact").textContent = "It just ran in " + ms.toFixed(1) + " ms on your device.";
    $("speedLine").textContent = "The model ran in " + ms.toFixed(1) + " ms on your device.";
  }

  // ---------- start ----------

  clips.forEach((clip, i) => {
    const option = document.createElement("option");
    option.value = i; option.textContent = clip.label;
    $("goal").appendChild(option);
  });
  fillResults();
  window.addEventListener("resize", resize);
  const first = Math.max(0, clips.findIndex((c) => c.name === (DATA.firstClip || "")));
  setClip(first);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw);
  setTimeout(timeModel, 300);
})();
