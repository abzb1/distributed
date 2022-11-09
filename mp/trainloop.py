import time

import torch as torch
from torch import distributed as dist


def mp_train_loop_first(rank: int,
                        model,
                        train_dataloader,
                        optimizer,
                        scheduler,
                        epochs: int,
                        split_size: int,
                        num_split: int,
                        output_size: list):

    backward_recv_buff_lst = []
    micro_output_size = output_size
    micro_output_size[0] = split_size
    micro_output_size = tuple(micro_output_size)

    for i in range(num_split):
        backward_recv_buff_lst.append(torch.empty(micro_output_size).cuda())

    for t in range(epochs):
        print(f"rank {rank} epoch {t} start")
        start = time.time()

        for batch, (X, _) in enumerate(train_dataloader):
            pred_buff_lst = []
            optimizer.zero_grad()

            for i, X in enumerate(iter(X.split(split_size))):
                model1_in = X.to("cuda")
                pred_buff_lst.append(model(model1_in))
                dist.isend(pred_buff_lst[i], rank+1)
            
            for i in range(num_split-1, -1, -1):
                dist.recv(backward_recv_buff_lst[i], rank+1)
                pred_buff_lst[i].backward(backward_recv_buff_lst[i])

            optimizer.step()

        scheduler.step()

        end = time.time()
        total = end-start
        print(f"{rank} {t} th epoch {total:>3f} s elapsed")
        print(f"ETA : {total*(epochs-t-1)} s")
        
    torch.save(model, "split_"+str(num_split)+"_model_"+str(rank)+".pth")

def mp_train_loop_middle(rank: int,
                        model,
                        train_dataloader,
                        optimizer,
                        scheduler,
                        epochs: int,
                        split_size: int,
                        num_split: int,
                        input_size: list,
                        output_size: list):

    forward_recv_buff_lst = []
    micro_input_size = input_size
    micro_input_size[0] = split_size
    micro_input_size = tuple(micro_input_size)

    backward_recv_buff_lst = []
    micro_output_size = output_size
    micro_output_size[0] = split_size
    micro_output_size = tuple(micro_output_size)

    for i in range(num_split):
        forward_recv_buff_lst.append(torch.empty(micro_input_size).cuda())
        backward_recv_buff_lst.append(torch.empty(micro_output_size).cuda())

    for t in range(epochs):
        print(f"rank {rank} epoch {t} start")
        start = time.time()

        for batch, _ in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_buff_lst = []
            pred_buff_lst = []

            for i in range(num_split):
                dist.recv(forward_recv_buff_lst[i], rank-1)
                model_in = forward_recv_buff_lst[i].clone().detach().requires_grad_(True)
                input_buff_lst.append(model_in)

                pred_buff_lst.append(model(model_in))
                dist.isend(pred_buff_lst[i], rank+1)
            
            # Pipelined Backward
            for i in range(num_split-1, -1, -1):
                dist.recv(backward_recv_buff_lst[i], rank+1)
                pred_buff_lst[i].backward(backward_recv_buff_lst[i])
                dist.isend(input_buff_lst[i].grad, rank-1)

            optimizer.step()

        end = time.time()
        total = end-start

        print(f"{rank} {t} th epoch {total:>3f} s elapsed")

        scheduler.step()

    torch.save(model, "split_"+str(num_split)+"_model_"+str(rank)+".pth")
    
def mp_train_loop_last(rank: int,
                       model,
                       loss_fn,
                       train_dataloader,
                       optimizer,
                       scheduler,
                       epochs: int,
                       split_size: int,
                       num_split: int,
                       input_size: list):

    forward_recv_buff_lst = []
    micro_input_size = input_size
    micro_input_size[0] = split_size
    micro_input_size = tuple(micro_input_size)

    for i in range(num_split):
        forward_recv_buff_lst.append(torch.empty(micro_input_size).cuda())

    for t in range(epochs):
        print(f"rank {rank} epoch {t} start")

        for batch, (_, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_buff_lst = []
            loss_lst = []

            for i, y in enumerate(iter(y.split(split_size))):
                label = y.to("cuda")
                dist.recv(forward_recv_buff_lst[i], rank-1)
                input_buff_lst.append(forward_recv_buff_lst[i].clone().detach().requires_grad_(True))

                pred = model(input_buff_lst[i])
                loss = loss_fn(pred, label)
                loss = loss/num_split
                loss_lst.append(loss)
                # print(f"{rank} {t} epoch {batch} batch {batch} micro batch {i} loss: {loss}")
            
            for i in range(num_split-1, -1, -1):
                input_buff_lst[i].retain_grad()
                loss_lst[i].backward()
                dist.isend(input_buff_lst[i].grad, rank-1)
            
            if batch % 100 == 0:
                if batch == 0:
                    continue
                print(f"{rank}, {t} epoch {batch} batch")

            optimizer.step()

        scheduler.step()

    torch.save(model, "split_"+str(num_split)+"_model_"+str(rank)+".pth")

def mp_train_loop_first_prof(rank: int,
                        model,
                        train_dataloader,
                        optimizer,
                        scheduler,
                        epochs: int,
                        split_size: int,
                        num_split: int,
                        output_size: list,
                        batch_size,
                        world_size):

    backward_recv_buff_lst = []
    micro_output_size = output_size
    micro_output_size[0] = split_size
    micro_output_size = tuple(micro_output_size)

    for i in range(num_split):
        backward_recv_buff_lst.append(torch.empty(micro_output_size).cuda())

    for t in range(epochs):
        print(f"rank {rank} epoch {t} start")
        start = time.time()

        with torch.profiler.profile(schedule=torch.profiler.\
                                             schedule(wait=100,\
                                                      warmup=100,\
                                                      active=10,\
                                                      repeat=1,\
                                                      skip_first=100),
                                    on_trace_ready = torch.profiler.\
                                                     tensorboard_trace_handler("./prof/batchsize_"+\
                                                                               str(batch_size)+\
                                                                               "_worldsize_"+str(world_size)),
                                    record_shapes = True,
                                    profile_memory = True,
                                    with_flops = True,
                                    with_stack = True
                                    ) as prof:                

            for batch, (X, _) in enumerate(train_dataloader):
                pred_buff_lst = []
                if optimizer is not None:
                    optimizer.zero_grad()
                back_request_buff_lst=[]

                for i, X in enumerate(iter(X.split(split_size))):
                    model1_in = X.to("cuda")
                    pred_buff_lst.append(model(model1_in))
                    dist.isend(pred_buff_lst[i], rank+1)
                
                back_request_buff_lst.append(dist.irecv(backward_recv_buff_lst[0], rank+1))
                back_request_buff_lst[0].wait()
                pred_buff_lst[0].backward(backward_recv_buff_lst[0])

                # Pipelined Backward
                for i in range(1, num_split):
                    back_request_buff_lst.append(dist.irecv(backward_recv_buff_lst[i], rank+1))

                for i in range(1, num_split):
                    back_request_buff_lst[i].wait()
                    pred_buff_lst[i].backward(backward_recv_buff_lst[i])

                if optimizer is not None:
                    optimizer.step()
                prof.step()
        if scheduler is not None:
            scheduler.step()

        end = time.time()
        total = end-start
        print(f"{rank} {t} th epoch {total:>3f} s elapsed")
        print(f"ETA : {total*(epochs-t-1)} s")
        
    torch.save(model, "split_"+str(num_split)+"_model_"+str(rank)+".pth")

def mp_train_loop_middle_prof(rank: int,
                        model,
                        train_dataloader,
                        optimizer,
                        scheduler,
                        epochs: int,
                        split_size: int,
                        num_split: int,
                        input_size: list,
                        output_size: list,
                        batch_size,
                        world_size):

    forward_recv_buff_lst = []
    micro_input_size = input_size
    micro_input_size[0] = split_size
    micro_input_size = tuple(micro_input_size)

    backward_recv_buff_lst = []
    micro_output_size = output_size
    micro_output_size[0] = split_size
    micro_output_size = tuple(micro_output_size)

    for i in range(num_split):
        forward_recv_buff_lst.append(torch.empty(micro_input_size).cuda())
        backward_recv_buff_lst.append(torch.empty(micro_output_size).cuda())

    for t in range(epochs):
        print(f"rank {rank} epoch {t} start")
        start = time.time()

        with torch.profiler.profile(schedule=torch.profiler.\
                                             schedule(wait=100,\
                                                      warmup=100,\
                                                      active=10,\
                                                      repeat=1,\
                                                      skip_first=100),
                                    on_trace_ready = torch.profiler.\
                                                     tensorboard_trace_handler("./prof/batchsize_"+\
                                                                               str(batch_size)+\
                                                                               "_worldsize_"+str(world_size)),
                                    record_shapes = True,
                                    profile_memory = True,
                                    with_flops = True,
                                    with_stack = True
                                    ) as prof:        

            for batch, _ in enumerate(train_dataloader):
                if optimizer is not None:
                    optimizer.zero_grad()
                input_buff_lst = []
                pred_buff_lst = []
                for_request_buff_lst = []
                back_request_buff_lst=[]

                for i in range(num_split):
                    for_request_buff_lst.append(dist.irecv(forward_recv_buff_lst[i], rank-1))

                for i in range(num_split):
                    for_request_buff_lst[i].wait()
                    model_in = forward_recv_buff_lst[i].clone().detach().requires_grad_(True)
                    input_buff_lst.append(model_in)

                    pred_buff_lst.append(model(model_in))
                    dist.isend(pred_buff_lst[i], rank+1)

                back_request_buff_lst.append(dist.irecv(backward_recv_buff_lst[0], rank+1))
                back_request_buff_lst[0].wait()
                pred_buff_lst[0].backward(backward_recv_buff_lst[0])
                dist.isend(input_buff_lst[0].grad,rank-1)

                # Pipelined Backward
                for i in range(1, num_split):
                    back_request_buff_lst.append(dist.irecv(backward_recv_buff_lst[i], rank+1))

                for i in range(1, num_split):
                    back_request_buff_lst[i].wait()
                    pred_buff_lst[i].backward(backward_recv_buff_lst[i])
                    dist.isend(input_buff_lst[i].grad,rank-1)

                if optimizer is not None:
                    optimizer.step()
                prof.step()

        end = time.time()
        total = end-start

        print(f"{rank} {t} th epoch {total:>3f} s elapsed")

        if scheduler is not None:
            scheduler.step()

    torch.save(model, "split_"+str(num_split)+"_model_"+str(rank)+".pth")
    
def mp_train_loop_last_prof(rank: int,
                       model,
                       loss_fn,
                       train_dataloader,
                       optimizer,
                       scheduler,
                       epochs: int,
                       split_size: int,
                       num_split: int,
                       input_size: list,
                       batch_size,
                       world_size):

    forward_recv_buff_lst = []
    micro_input_size = input_size
    micro_input_size[0] = split_size
    micro_input_size = tuple(micro_input_size)

    for i in range(num_split):
        forward_recv_buff_lst.append(torch.empty(micro_input_size).cuda())

    for t in range(epochs):
        print(f"rank {rank} epoch {t} start")

        with torch.profiler.profile(schedule=torch.profiler.\
                                             schedule(wait=100,\
                                                      warmup=100,\
                                                      active=10,\
                                                      repeat=1,\
                                                      skip_first=100),
                                    on_trace_ready = torch.profiler.\
                                                     tensorboard_trace_handler("./prof/intra_ddp_imageNet_VGGD_batchsize_"+\
                                                                               str(batch_size)+\
                                                                               "_worldsize_"+str(world_size)),
                                    record_shapes = True,
                                    profile_memory = True,
                                    with_flops = True,
                                    with_stack = True
                                    ) as prof:

            for batch, (_, y) in enumerate(train_dataloader):
                if optimizer is not None:
                    optimizer.zero_grad()
                input_buff_lst = []
                loss_lst = []
                for_request_buff_lst = []

                for i in range(num_split):
                    for_request_buff_lst.append(dist.irecv(forward_recv_buff_lst[i], rank-1))

                for i, y in enumerate(iter(y.split(split_size))):
                    label = y.to("cuda")
                    for_request_buff_lst[i].wait()
                    input_buff_lst.append(forward_recv_buff_lst[i].clone().detach().requires_grad_(True))

                    pred = model(input_buff_lst[i])
                    loss = loss_fn(pred, label)
                    loss = loss/num_split
                    loss_lst.append(loss)
                    # print(f"{rank} {t} epoch {batch} batch {batch} micro batch {i} loss: {loss}")
                
                for i in range(num_split-1, -1, -1):
                    input_buff_lst[i].retain_grad()
                    loss_lst[i].backward()
                    dist.isend(input_buff_lst[i].grad, rank-1)
                
                if optimizer is not None:
                    optimizer.step()
                prof.step()
            
                if batch % 100 == 0:
                    if batch == 0:
                        continue
                    print(f"{rank}, {t} epoch {batch} batch")

        if scheduler is not None:
            scheduler.step()

    torch.save(model, "split_"+str(num_split)+"_model_"+str(rank)+".pth")
