# Change3D Integration

This project entry integrates the external `Change3D` repository with this
Open-CD workspace without rewriting the model into MMEngine.

Default external root:

```text
D:\Projects\change_detection\Change3D
```

You can override it with `--change3d-root` or the `CHANGE3D_ROOT`
environment variable.

## Binary Change Detection

Train:

```bash
python projects/change3d/run_change3d.py train-bcd ^
  --file-root D:\Projects\change_detection\CD_datasets\CD_mine\11 ^
  --work-dir results\change3d\bcd ^
  --max-steps 40000 ^
  --batch-size 4 ^
  --num-workers 0
```

Large TIF inference:

```bash
python projects/change3d/run_change3d.py infer-bcd ^
  --tif1 path\to\time1.tif ^
  --tif2 path\to\time2.tif ^
  --pt-path path\to\bcd_model.pt ^
  --out-dir results\change3d\bcd_infer
```

## Multi-Class / All-Element Change Detection

Train:

```bash
python projects/change3d/run_change3d.py train-mcd ^
  --file-root path\to\mcd_dataset ^
  --work-dir results\change3d\mcd ^
  --num-class 31 ^
  --max-steps 80000 ^
  --batch-size 2 ^
  --num-workers 0
```

Large TIF inference:

```bash
python projects/change3d/run_change3d.py infer-mcd ^
  --tif1 path\to\time1.tif ^
  --tif2 path\to\time2.tif ^
  --pt-path path\to\mcd_model.pt ^
  --out-dir results\change3d\mcd_infer ^
  --num-classes 31
```

The external model code still lives in `Change3D`; this wrapper only gives
Open-CD a stable entry point for launching its training and inference workflows.
