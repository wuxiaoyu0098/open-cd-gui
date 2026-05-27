# Change3D multi-class / all-element change detection config.
#
# This config is dispatched by tools/train.py and tools/test.py to
# projects/change3d/run_change3d.py.

work_dir = 'results/change3d/mcd'

change3d = dict(
    task='mcd',
    file_root=r'D:\Projects\change_detection\CD_datasets\CD_mine\11',
    dataset='Change3D-MCD',
    im1_dir='A',
    im2_dir='B',
    label_dir='label',
    in_height=512,
    in_width=512,
    num_class=31,
    ignore_index=255,
    max_steps=None,
    max_epochs=120,
    batch_size=2,
    num_workers=0,
    lr=2e-4,
    pretrained='model/X3D_L.pyth',
    gpu_id=0,
    devices=1,
    accelerator='gpu',
    strategy='auto',
    precision=16,
    val_check_interval=1.0,
    check_val_every_n_epoch=1,
    train_log_interval=10,
    class_weight_mode='median_freq_nonzero',
    dice_weight=1.0,
    infer=dict(
        tif1='',
        tif2='',
        out_dir='results/change3d/mcd_infer',
        out_name='change3d_mcd',
        tile_size=512,
        overlap=128,
        batch_size=8,
        num_classes=31,
        export_class_ids='',
        remove_holes_area=200000,
        remove_objects_min_size=800,
        min_area_mu=0.1,
        memmap_dir='',
        gpu_id=0,
        device='cuda',
    ))
