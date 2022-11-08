torchrun \
--standalone \
--nnodes=1 \
--nproc_per_node=2 \
pipe_mp.py -w 2 \
--model VGGD \
--dataset imagenet \
-e 1