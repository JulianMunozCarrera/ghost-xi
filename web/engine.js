/*
  Ghost XI engine. The same math as the Python code, written in JavaScript
  so the model can run inside the web page with nothing to install.

  It mirrors these Python files:
    ghostxi/tokens.py        -> buildTokens
    ghostxi/model_numpy.py   -> Model
    ghostxi/pitch.py         -> speedFromPositions
    ghostxi/ghost.py         -> ghostsAt, teamGhost, soloGhosts
    ghostxi/analysis.py      -> ghostDefense
    ghostxi/danger.py        -> danger, attackControl, space

  A TV camera does not show every player. A position of NaN means
  "not on screen in this frame". Every function here copes with that.

  A test (tests/test_web_engine.py) runs this file with Node and checks
  that it gives the same numbers as Python.
*/
(function (root) {
  "use strict";

  const HALF_LENGTH = 52.5;
  const HALF_WIDTH = 34;
  const GOAL_WIDTH = 7.32;
  const MAX_PLAYERS = 11;
  const TOKENS = 23;
  const FEATURES = 21;
  const FIRST_ATTACKER = 1;
  const FIRST_DEFENDER = 12;
  const POSITION_SCALE = 50;
  const SPEED_SCALE = 8;
  const TIME_SCALE = 10;
  const MOVE_SCALE = 10;
  const NEAR_SCALE = 20;
  const MAX_SECONDS = 10;
  const HEADS = 4;

  // ---------- weights ----------

  // The weights travel as one long base64 string. The manifest says which
  // slice of it belongs to which part of the model.
  function decodeWeights(base64, manifest) {
    const text = typeof atob === "function" ? atob(base64) : Buffer.from(base64, "base64").toString("binary");
    const bytes = new Uint8Array(text.length);
    for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
    const all = new Float32Array(bytes.buffer);
    const weights = {};
    let at = 0;
    for (const [name, shape] of manifest) {
      const size = shape.reduce((a, b) => a * b, 1);
      weights[name] = all.subarray(at, at + size);
      at += size;
    }
    return weights;
  }

  // ---------- model ----------

  // y = x * W + b for a list of tokens. W is stored row by row, one row per output.
  function linear(x, count, sizeIn, w, b, sizeOut) {
    const y = new Float32Array(count * sizeOut);
    for (let i = 0; i < count; i++) {
      const xi = i * sizeIn;
      for (let o = 0; o < sizeOut; o++) {
        let sum = b[o];
        const wo = o * sizeIn;
        for (let k = 0; k < sizeIn; k++) sum += w[wo + k] * x[xi + k];
        y[i * sizeOut + o] = sum;
      }
    }
    return y;
  }

  // Shift and scale each token so its numbers have mean 0 and spread 1.
  function layerNorm(x, count, size, weight, bias) {
    const y = new Float32Array(count * size);
    for (let i = 0; i < count; i++) {
      let mean = 0;
      for (let k = 0; k < size; k++) mean += x[i * size + k];
      mean /= size;
      let spread = 0;
      for (let k = 0; k < size; k++) spread += (x[i * size + k] - mean) ** 2;
      spread = Math.sqrt(spread / size + 1e-5);
      for (let k = 0; k < size; k++) y[i * size + k] = ((x[i * size + k] - mean) / spread) * weight[k] + bias[k];
    }
    return y;
  }

  class Model {
    constructor(weights) {
      this.w = weights;
      this.width = weights["read.bias"].length;
      this.hidden = weights["layers.0.think1.bias"].length;
      this.layers = 0;
      while (weights["layers." + this.layers + ".norm1.weight"]) this.layers++;
    }

    // tokens: Float32Array of count * 21 numbers. Only real tokens, no empty slots.
    // Returns count * 2 numbers: how far each token moved, in units of 10 m.
    run(tokens, count) {
      const w = this.w, width = this.width, perHead = width / HEADS;
      let x = linear(tokens, count, FEATURES, w["read.weight"], w["read.bias"], width);

      for (let l = 0; l < this.layers; l++) {
        const p = "layers." + l + ".";
        // Part 1: every token looks at every other token.
        let h = layerNorm(x, count, width, w[p + "norm1.weight"], w[p + "norm1.bias"]);
        const q = linear(h, count, width, w[p + "query.weight"], w[p + "query.bias"], width);
        const k = linear(h, count, width, w[p + "key.weight"], w[p + "key.bias"], width);
        const v = linear(h, count, width, w[p + "value.weight"], w[p + "value.bias"], width);
        const attended = new Float32Array(count * width);
        const scores = new Float64Array(count);
        for (let head = 0; head < HEADS; head++) {
          const off = head * perHead;
          for (let i = 0; i < count; i++) {
            let top = -Infinity;
            for (let j = 0; j < count; j++) {
              let s = 0;
              for (let d = 0; d < perHead; d++) s += q[i * width + off + d] * k[j * width + off + d];
              s /= Math.sqrt(perHead);
              scores[j] = s;
              if (s > top) top = s;
            }
            let total = 0;
            for (let j = 0; j < count; j++) { scores[j] = Math.exp(scores[j] - top); total += scores[j]; }
            for (let j = 0; j < count; j++) {
              const share = scores[j] / total;
              for (let d = 0; d < perHead; d++) attended[i * width + off + d] += share * v[j * width + off + d];
            }
          }
        }
        const mixed = linear(attended, count, width, w[p + "mix.weight"], w[p + "mix.bias"], width);
        for (let i = 0; i < x.length; i++) x[i] += mixed[i];

        // Part 2: each token processes what it gathered.
        h = layerNorm(x, count, width, w[p + "norm2.weight"], w[p + "norm2.bias"]);
        const t1 = linear(h, count, width, w[p + "think1.weight"], w[p + "think1.bias"], this.hidden);
        for (let i = 0; i < t1.length; i++) if (t1[i] < 0) t1[i] = 0;
        const t2 = linear(t1, count, this.hidden, w[p + "think2.weight"], w[p + "think2.bias"], width);
        for (let i = 0; i < x.length; i++) x[i] += t2[i];
      }

      x = layerNorm(x, count, width, w["norm.weight"], w["norm.bias"]);
      return linear(x, count, width, w["answer.weight"], w["answer.bias"], 2);
    }
  }

  // ---------- clips ----------

  // Clips are stored as whole centimeters to keep the page small. Turn them into meters.
  // A null in the list means the player was not on screen. That becomes NaN.
  function unpackClip(raw) {
    const toMeters = (list) => { const out = new Float32Array(list.length); for (let i = 0; i < list.length; i++) out[i] = list[i] === null ? NaN : list[i] / 100; return out; };
    const clip = Object.assign({}, raw);
    clip.ball = toMeters(raw.ball);
    clip.att = toMeters(raw.att);
    clip.def = toMeters(raw.def);
    clip.reach = Math.max(1, Math.round(0.12 * clip.fps));   // same rule as speed_reach in samples.py
    clip.halfSecond = Math.max(1, Math.round(0.5 * clip.fps));
    clip.ballV = speedFromPositions(clip.ball, clip.frames, 1, clip.fps, clip.reach, 40);
    clip.attV = speedFromPositions(clip.att, clip.frames, clip.nAtt, clip.fps, clip.reach, 12);
    clip.defV = speedFromPositions(clip.def, clip.frames, clip.nDef, clip.fps, clip.reach, 12);
    return clip;
  }

  // Speed from positions: compare the position a few frames ahead with a few frames back.
  // xy holds frames * players * 2 numbers. If one side is missing we use the
  // other side. If we cannot tell, the speed is 0.
  function speedFromPositions(xy, frames, players, fps, reach, topSpeed) {
    const out = new Float32Array(xy.length);
    for (let f = 0; f < frames; f++) {
      for (let p = 0; p < players; p++) {
        const i = (f * players + p) * 2;
        const a = ((f + reach) * players + p) * 2, b = ((f - reach) * players + p) * 2;
        const here = !Number.isNaN(xy[i]);
        const hasAhead = f + reach < frames && !Number.isNaN(xy[a]);
        const hasBack = f - reach >= 0 && !Number.isNaN(xy[b]);
        let vx = 0, vy = 0;
        if (hasAhead && hasBack) { vx = (xy[a] - xy[b]) / (2 * reach / fps); vy = (xy[a + 1] - xy[b + 1]) / (2 * reach / fps); }
        else if (hasAhead && here) { vx = (xy[a] - xy[i]) / (reach / fps); vy = (xy[a + 1] - xy[i + 1]) / (reach / fps); }
        else if (hasBack && here) { vx = (xy[i] - xy[b]) / (reach / fps); vy = (xy[i + 1] - xy[b + 1]) / (reach / fps); }
        const speed = Math.hypot(vx, vy);
        if (speed > topSpeed) { vx *= topSpeed / speed; vy *= topSpeed / speed; }
        out[i] = vx; out[i + 1] = vy;
      }
    }
    return out;
  }

  // Everything about one frame of a clip. Copies, so the caller may change them.
  function momentOf(clip, frame) {
    return {
      ball: clip.ball.slice(frame * 2, frame * 2 + 2),
      ballV: clip.ballV.slice(frame * 2, frame * 2 + 2),
      att: clip.att.slice(frame * clip.nAtt * 2, (frame + 1) * clip.nAtt * 2),
      attV: clip.attV.slice(frame * clip.nAtt * 2, (frame + 1) * clip.nAtt * 2),
      def: clip.def.slice(frame * clip.nDef * 2, (frame + 1) * clip.nDef * 2),
      defV: clip.defV.slice(frame * clip.nDef * 2, (frame + 1) * clip.nDef * 2),
      attKeeper: clip.attKeeper,
      defKeeper: clip.defKeeper,
    };
  }

  // ---------- tokens ----------

  // The 21 numbers for one ball or player. Same layout as ghostxi/tokens.py.
  function oneToken(out, at, kind, isKeeper, hidden, nowXY, nowV, thenXY, thenV, i, seconds, ballNow, ballThen, thenMissing) {
    out[at + kind] = 1;
    out[at + 3] = isKeeper ? 1 : 0;
    out[at + 4] = hidden ? 1 : 0;
    if (!hidden) {
      out[at + 5] = nowXY[i] / POSITION_SCALE; out[at + 6] = nowXY[i + 1] / POSITION_SCALE;
      out[at + 7] = nowV[i] / SPEED_SCALE; out[at + 8] = nowV[i + 1] / SPEED_SCALE;
    }
    if (!thenMissing) {
      out[at + 9] = thenXY[i] / POSITION_SCALE; out[at + 10] = thenXY[i + 1] / POSITION_SCALE;
      out[at + 11] = thenV[i] / SPEED_SCALE; out[at + 12] = thenV[i + 1] / SPEED_SCALE;
    }
    out[at + 13] = seconds / TIME_SCALE;
    // Six numbers about the ball.
    out[at + 14] = (ballNow[0] - ballThen[0]) / NEAR_SCALE; out[at + 15] = (ballNow[1] - ballThen[1]) / NEAR_SCALE;
    if (kind !== 0) {
      if (!hidden) {
        out[at + 16] = (nowXY[i] - ballNow[0]) / NEAR_SCALE; out[at + 17] = (nowXY[i + 1] - ballNow[1]) / NEAR_SCALE;
      }
      if (!thenMissing) {
        out[at + 18] = (thenXY[i] - ballThen[0]) / NEAR_SCALE; out[at + 19] = (thenXY[i + 1] - ballThen[1]) / NEAR_SCALE;
      }
    }
    out[at + 20] = thenMissing ? 1 : 0;
  }

  // Build the token list for one moment. Returns the tokens (only real ones)
  // and, for each defender, which token is his (-1 if he has none).
  // A hidden defender needs his "then". Everyone else needs his "now".
  function buildTokens(now, then, seconds, hidden) {
    const nAtt = now.att.length / 2, nDef = now.def.length / 2;
    const tokens = new Float32Array((1 + nAtt + nDef) * FEATURES);
    const defToken = new Array(nDef).fill(-1);
    let count = 0;
    oneToken(tokens, 0, 0, false, false, now.ball, now.ballV, then.ball, then.ballV, 0, seconds, now.ball, then.ball, false);
    count++;
    for (let i = 0; i < nAtt; i++) {
      if (Number.isNaN(now.att[i * 2])) continue;
      oneToken(tokens, count * FEATURES, 1, i === now.attKeeper, false, now.att, now.attV, then.att, then.attV, i * 2, seconds, now.ball, then.ball, Number.isNaN(then.att[i * 2]));
      count++;
    }
    for (let i = 0; i < nDef; i++) {
      const seenThen = !Number.isNaN(then.def[i * 2]), seenNow = !Number.isNaN(now.def[i * 2]);
      if ((hidden[i] && !seenThen) || (!hidden[i] && !seenNow)) continue;
      oneToken(tokens, count * FEATURES, 2, i === now.defKeeper, hidden[i], now.def, now.defV, then.def, then.defV, i * 2, seconds, now.ball, then.ball, !seenThen);
      defToken[i] = count;
      count++;
    }
    return { tokens: tokens.subarray(0, count * FEATURES), count, defToken };
  }

  // ---------- ghosts ----------

  // Which defenders can get a ghost: the ones who were on screen "then".
  function canGhost(then) {
    const out = [];
    for (let i = 0; i < then.def.length / 2; i++) out.push(!Number.isNaN(then.def[i * 2]));
    return out;
  }

  // Where should the hidden defenders be "now"? Returns nDef * 2 numbers.
  // Everyone who is not hidden (or cannot be ghosted) is NaN in the answer.
  function ghostsAt(model, now, then, seconds, hidden) {
    const nDef = now.def.length / 2;
    const out = new Float32Array(nDef * 2).fill(NaN);
    if (seconds <= 0 || Number.isNaN(now.ball[0]) || Number.isNaN(then.ball[0])) return out;
    const built = buildTokens(now, then, seconds, hidden);
    const answer = model.run(built.tokens, built.count);
    for (let i = 0; i < nDef; i++) {
      if (!hidden[i] || built.defToken[i] < 0) continue;
      const t = built.defToken[i] * 2;
      out[i * 2] = then.def[i * 2] + answer[t] * MOVE_SCALE;
      out[i * 2 + 1] = then.def[i * 2 + 1] + answer[t + 1] * MOVE_SCALE;
    }
    return out;
  }

  // Team ghost: hide every defender we can at once.
  function teamGhost(model, now, then, seconds) {
    return ghostsAt(model, now, then, seconds, canGhost(then));
  }

  // Solo ghosts: hide one defender at a time, the others stay real.
  function soloGhosts(model, now, then, seconds) {
    const nDef = now.def.length / 2;
    const out = new Float32Array(nDef * 2).fill(NaN);
    const can = canGhost(then);
    for (let d = 0; d < nDef; d++) {
      if (!can[d]) continue;
      const hidden = new Array(nDef).fill(false);
      hidden[d] = true;
      const one = ghostsAt(model, now, then, seconds, hidden);
      out[d * 2] = one[d * 2]; out[d * 2 + 1] = one[d * 2 + 1];
    }
    return out;
  }

  // How fast the ghosts move, from where they are now and where they were
  // a little earlier. Capped at 12 m/s like everything else. No ghost
  // earlier means we treat him as standing still.
  function ghostSpeed(ghostsNow, ghostsEarlier, secondsApart) {
    const out = new Float32Array(ghostsNow.length);
    if (secondsApart <= 0) return out;
    for (let i = 0; i < ghostsNow.length; i += 2) {
      let vx = (ghostsNow[i] - ghostsEarlier[i]) / secondsApart, vy = (ghostsNow[i + 1] - ghostsEarlier[i + 1]) / secondsApart;
      if (Number.isNaN(vx) || Number.isNaN(vy)) { vx = 0; vy = 0; }
      const speed = Math.hypot(vx, vy);
      if (speed > 12) { vx *= 12 / speed; vy *= 12 / speed; }
      out[i] = vx; out[i + 1] = vy;
    }
    return out;
  }

  // The defense with ghosts swapped in. A defender stands on his ghost if he
  // has one and is on screen now. Otherwise he stays where he really is.
  function ghostDefense(now, ghosts, ghostV) {
    const pos = new Float32Array(now.def), vel = new Float32Array(now.defV);
    const swapped = [];
    for (let i = 0; i < now.def.length / 2; i++) {
      const swap = !Number.isNaN(now.def[i * 2]) && !Number.isNaN(ghosts[i * 2]);
      swapped.push(swap);
      if (!swap) continue;
      pos[i * 2] = ghosts[i * 2]; pos[i * 2 + 1] = ghosts[i * 2 + 1];
      vel[i * 2] = ghostV[i * 2]; vel[i * 2 + 1] = ghostV[i * 2 + 1];
    }
    return { pos, vel, swapped };
  }

  // ---------- danger ----------

  const REACTION = 0.7, TOP_SPEED = 5.0, FUZZ = 0.45, REACH = 24.0;

  // A grid of spots in front of the goal, and how much each one counts.
  const ZONE = (function () {
    const spots = [], weight = [];
    for (let x = HALF_LENGTH - REACH + 0.5; x < HALF_LENGTH; x += 1) {
      for (let y = -REACH + 0.5; y < REACH; y += 1) {
        const dx = HALF_LENGTH - x;
        if (Math.hypot(dx, y) > REACH) continue;
        spots.push(x, y);
        weight.push(Math.atan2(GOAL_WIDTH * dx, dx * dx + y * y - (GOAL_WIDTH / 2) ** 2));
      }
    }
    return { spots: new Float32Array(spots), weight: new Float32Array(weight), count: weight.length };
  })();

  // The soonest any of these players can reach a spot. Players who are not on screen are skipped.
  function firstArrival(x, y, players, speeds) {
    let best = Infinity;
    for (let i = 0; i < players.length; i += 2) {
      if (Number.isNaN(players[i])) continue;
      const px = players[i] + speeds[i] * REACTION, py = players[i + 1] + speeds[i + 1] * REACTION;
      const t = REACTION + Math.hypot(px - x, py - y) / TOP_SPEED;
      if (t < best) best = t;
    }
    return best;
  }

  // For each spot in the zone: the chance an attacker gets there first.
  function attackControl(att, attV, def, defV) {
    const out = new Float32Array(ZONE.count);
    for (let s = 0; s < ZONE.count; s++) {
      const x = ZONE.spots[s * 2], y = ZONE.spots[s * 2 + 1];
      const a = firstArrival(x, y, att, attV), d = firstArrival(x, y, def, defV);
      // No attacker on screen: nothing is dangerous. No defender: everything is.
      out[s] = a === Infinity ? 0 : d === Infinity ? 1 : 1 / (1 + Math.exp(-(d - a) / FUZZ));
    }
    return out;
  }

  function danger(att, attV, def, defV) {
    const control = attackControl(att, attV, def, defV);
    let top = 0, bottom = 0;
    for (let s = 0; s < ZONE.count; s++) { top += control[s] * ZONE.weight[s]; bottom += ZONE.weight[s]; }
    return top / bottom;
  }

  // Distance from the ball to the nearest outfield defender on screen. NaN if there is none.
  function space(ball, def, keeper) {
    let best = Infinity;
    for (let i = 0; i < def.length / 2; i++) {
      if (i === keeper || Number.isNaN(def[i * 2])) continue;
      best = Math.min(best, Math.hypot(def[i * 2] - ball[0], def[i * 2 + 1] - ball[1]));
    }
    return best === Infinity ? NaN : best;
  }

  const api = {
    HALF_LENGTH, HALF_WIDTH, GOAL_WIDTH, MAX_SECONDS, ZONE,
    decodeWeights, Model, unpackClip, speedFromPositions, momentOf, buildTokens,
    canGhost, ghostsAt, teamGhost, soloGhosts, ghostSpeed, ghostDefense, attackControl, danger, space,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.GhostEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
