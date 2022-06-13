srun \
--container-image=/netscratch/hufe/gan_image_fixed.sqsh \
--container-mounts=/netscratch/hufe/gan:/netscratch,./disentangling-stylegan2:/workspace \
--gpus=1 \
--mem=24G \
python lorenz_run.py