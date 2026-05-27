# Change3D binary change detection config.
#
# This config is dispatched by tools/train.py and tools/test.py to
# projects/change3d/run_change3d.py.

work_dir = 'results/change3d/bcd'

change3d = dict(
    task='bcd',
    file_root=r'D:\Projects\change_detection\CD_datasets\CD_mine\11',
    dataset='Change3D-CD',
    pre_dir='A',
    post_dir='B',
    label_dir='label',
    in_height=512,
    in_width=512,
    max_steps=None,
    max_epochs=120,
    batch_size=4,
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
    infer=dict(
        tif1='',
        tif2='',
        out_dir='results/change3d/bcd_infer',
        out_name='change3d_bcd',
        tile_size=512,
        overlap=128,
        batch_size=8,
        prob_threshold=0.5,
        remove_holes_area=200000,
        remove_objects_min_size=800,
        min_area_mu=0.1,
        memmap_dir='',
        gpu_id=0,
        device='cuda',
    ))
