// Runs the website's engine with Node and compares it with Python's answers.
// Used by tests/test_web_engine.py. Prints the biggest difference found.
const fs = require("fs");
const path = require("path");
const engine = require(path.join(__dirname, "..", "web", "engine.js"));
const data = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "web", "data.json")));
const expected = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "web", "expected.json")));

const model = new engine.Model(engine.decodeWeights(data.weights.data, data.weights.manifest));
let worst = 0;
const compare = (a, b) => { for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i])); };

for (const item of expected) {
  const clip = engine.unpackClip(data.clips[item.clip]);
  const then = engine.momentOf(clip, item.start);
  item.frames.forEach((frame, n) => {
    const now = engine.momentOf(clip, frame);
    const seconds = (frame - item.start) / clip.fps;
    compare(engine.teamGhost(model, now, then, seconds), item.team[n]);
    compare(engine.soloGhosts(model, now, then, seconds), item.solo[n]);
  });

  // The shot moment: off-by distances, danger and space.
  const shot = clip.shot, now = engine.momentOf(clip, shot);
  const seconds = (shot - item.start) / clip.fps;
  const solo = engine.soloGhosts(model, now, then, seconds);
  const offBy = [];
  for (let i = 0; i < clip.nDef; i++) offBy.push(Math.hypot(now.def[i * 2] - solo[i * 2], now.def[i * 2 + 1] - solo[i * 2 + 1]));
  compare(offBy, item.shot.off_by);

  const team = engine.teamGhost(model, now, then, seconds);
  // Ghost speed: compare with the ghosts half a second earlier (or the real defenders, at the start).
  const before = Math.max(shot - clip.halfSecond, 0);
  const earlier = before <= item.start ? engine.momentOf(clip, before).def
    : engine.teamGhost(model, engine.momentOf(clip, before), then, (before - item.start) / clip.fps);
  const ghostV = engine.ghostSpeed(team, earlier, (shot - before) / clip.fps);
  compare([engine.danger(now.att, now.attV, now.def, now.defV)], [item.shot.danger_real]);
  compare([engine.danger(now.att, now.attV, team, ghostV)], [item.shot.danger_ghost]);
  compare([engine.space(now.ball, now.def, clip.defKeeper)], [item.shot.space_real]);
  compare([engine.space(now.ball, team, clip.defKeeper)], [item.shot.space_ghost]);
}

// How long does one run of the model take?
const clip = engine.unpackClip(data.clips[0]);
const then = engine.momentOf(clip, 0), now = engine.momentOf(clip, clip.shot);
const began = process.hrtime.bigint();
for (let i = 0; i < 200; i++) engine.teamGhost(model, now, then, 5);
const ms = Number(process.hrtime.bigint() - began) / 1e6 / 200;

console.log(JSON.stringify({ worst_difference: worst, ms_per_run: ms }));
