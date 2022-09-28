torchrun \
--standalone \
--nnodes=1 \
--nproc_per_node=4 \
ddp.py -n 4
