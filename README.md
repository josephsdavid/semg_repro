# Reproduce paper

Code to reproduce [sEMG Gesture Recognition With a Simple Model of Attention](http://proceedings.mlr.press/v136/josephs20a)

## Reproduce environment

```bash
pip install -r requirements.txt
```

## Check code

```bash
cat repro.py
```

## Run experiment

Without IMU:

```bash
python repro.py
```

With IMU

```bash
python repro.py imu
```
