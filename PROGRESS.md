# Ghost XI: where we left off

Saved on 2026-11-10. The project is about half built. Nothing here is final yet.

## Done and working

- `ghostxi/data.py`: reads the 3 Metrica sample matches (both file formats), cuts them into defending spells.
- `ghostxi/model.py`: the ghost model (GhostNet) and the rollout code. Team mode and solo mode.
- `ghostxi/windows.py`, `ghostxi/baselines.py`: 5 second clips and the simple methods we compare against.
- `scripts/train.py`: trains 3 models, each tested on the match it never saw.
- `ghostxi/control.py`: pitch control "danger near goal" measure.
- `ghostxi/analysis.py`: cuts the play before a shot and runs the ghosts on it. Works on 8 of the 10 goals.
- `ghostxi/camera.py`, `ghostxi/render.py`, `ghostxi/vision.py`: draws TV style pictures from tracking data, finds the camera and the players again. First checks: camera off by about 0.1 m, players off by about 0.25 m, about 85 to 90% of players found.
- `scripts/evaluate.py`: written, checked on one match only.
- `scripts/vision_test.py`: written, NOT run yet.
- `web/template.html`: the website demo, written, NOT opened in a browser yet.

## Not done

1. Training was cut off. No final models are saved. Rerun it (about 40 minutes):
   `python scripts/train.py --data data/metrica/data`
2. Run `scripts/evaluate.py` and `scripts/vision_test.py` and fix whatever breaks.
3. Write `scripts/build_site.py`: run the goal analysis, put the data into `web/template.html`, save as `docs/index.html`.
4. Open the website and check it. Publish it.
5. Tests in `tests/`.
6. README with the real numbers, requirements.txt, LICENSE.
7. Check why 2 goals have no clip (Sample_Game_1 frame 19807, Sample_Game_2 frame 23002). Probably a penalty or free kick with under 2 s of open play.

## Early numbers (draft only, do not quote)

A quick 4 epoch trial, tested on Sample_Game_3, outfield defenders, metres off after 5 s:

- stand still 10.6, keep running 9.9, follow the ball 8.1, ghost model 4.1
- average over the 5 s: best simple method 4.0, ghost model 2.1

One early finding to be honest about: on that one match the ghost defence did NOT give up less danger than the real defence (shots: 35.2% real vs 35.4% ghost). The ghost is a typical defender, not a better one. The README and website must say this plainly unless the full run shows otherwise.

## Get the data

`git clone https://github.com/metrica-sports/sample-data data/metrica`
