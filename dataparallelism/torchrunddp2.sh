torchrun \
--standalone \
--nnodes=1 \
--nproc_per_node=2 \
ddp.py -n 2
