torchrun \
--standalone \
--nnodes=1 \
--nproc_per_node=8 \
ddp.py -n 8
