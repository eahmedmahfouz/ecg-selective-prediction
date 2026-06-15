Give it specs, not goals. "Implement temperature scaling: optimize scalar T via LBFGS on validation NLL, return calibrated probs as sigmoid(logits/T)" is good. "Make the model calibrated" is bad.
Review every diff before commit. Especially anything touching src/metrics.py or the conformal code. Subtle off-by-ones in calibration metrics will silently invalidate your whole paper.
Never let it run training. You launch training; Claude Code only writes the training code. This protects against accidentally wiping a checkpoint or running on the wrong fold.
Make it write tests. For every metric, every Predictor, ask for a unit test on synthetic data with known properties. Test-driven Claude Code is dramatically more reliable than vibes-driven Claude Code.
Keep CLAUDE.md updated. When you make a design decision (e.g. "we use 100Hz not 500Hz"), add it to CLAUDE.md. Future sessions will respect it.
Use git aggressively. Branch per method. If a session goes sideways, git reset --hard is your friend.
Don't let it touch the paper itself. Have it generate figures and tables from code. Write the prose yourself — your voice and reasoning are what reviewers evaluate, and AI-written methods sections read as AI-written.
