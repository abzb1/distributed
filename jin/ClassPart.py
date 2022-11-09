class PartTime:

    def __init__(self, for_compute_time: float, back_compute_time: float,
                 for_send_time: float, back_send_time: float,
                 num_split: int):
        
        self.num_split = num_split

        self.for_compute_time = for_compute_time
        
        self.back_compute_time = back_compute_time
        
        self.for_send_time = for_send_time
        self.back_send_time = back_send_time

        self.for_compute_start_time_lst=[0.0 for _ in range(self.num_split)]
        self.for_compute_end_time_lst=[0.0 for _ in range(self.num_split)]
        
        self.for_send_start_time_lst=[0.0 for _ in range(self.num_split)]
        self.for_send_end_time_lst=[0.0 for _ in range(self.num_split)]
        
        self.for_recv_start_time_lst=[0.0 for _ in range(self.num_split)]
        self.for_recv_end_time_lst=[0.0 for _ in range(self.num_split)]

        self.back_compute_start_time_lst=[0.0 for _ in range(self.num_split)]
        self.back_compute_end_time_lst=[0.0 for _ in range(self.num_split)]
        
        self.back_send_start_time_lst=[0.0 for _ in range(self.num_split)]
        self.back_send_end_time_lst=[0.0 for _ in range(self.num_split)]
        
        self.back_recv_start_time_lst=[0.0 for _ in range(self.num_split)]
        self.back_recv_end_time_lst=[0.0 for _ in range(self.num_split)]

    def get_for_compute_start_time_lst_elem(self, i):
        a = self.for_compute_start_time_lst[i]
        return a

    def set_for_compute_start_time_lst_elem(self, i, time):
        self.for_compute_start_time_lst[i] = time

    def get_for_compute_end_time_lst_elem(self, i):
        a = self.for_compute_end_time_lst[i]
        return a

    def set_for_compute_end_time_lst_elem(self, i, time):
        self.for_compute_end_time_lst[i] = time

    def get_for_send_start_time_lst_elem(self, i):
        a = self.for_send_start_time_lst[i]
        return a

    def set_for_send_start_time_lst_elem(self, i, time):
        self.for_send_start_time_lst[i] = time

    def get_for_send_end_time_lst_elem(self, i):
        a = self.for_send_end_time_lst[i]
        return a

    def set_for_send_end_time_lst_elem(self, i, time):
        self.for_send_end_time_lst[i] = time

    def get_for_recv_start_time_lst_elem(self, i):
        a = self.for_recv_start_time_lst[i]
        return a

    def set_for_recv_start_time_lst_elem(self, i, time):
        self.for_recv_start_time_lst[i] = time

    def get_for_recv_end_time_lst_elem(self, i):
        a = self.for_recv_end_time_lst[i]
        return a

    def set_for_recv_end_time_lst_elem(self, i, time):
        self.for_recv_end_time_lst[i] = time

    def get_back_compute_start_time_lst_elem(self, i):
        a = self.back_compute_start_time_lst[i]
        return a

    def set_back_compute_start_time_lst_elem(self, i, time):
        self.back_compute_start_time_lst[i] = time

    def get_back_compute_end_time_lst_elem(self, i):
        a = self.back_compute_end_time_lst[i]
        return a

    def set_back_compute_end_time_lst_elem(self, i, time):
        self.back_compute_end_time_lst[i] = time

    def get_back_send_start_time_lst_elem(self, i):
        a = self.back_send_start_time_lst[i]
        return a

    def set_back_send_start_time_lst_elem(self, i, time):
        self.back_send_start_time_lst[i] = time

    def get_back_send_end_time_lst_elem(self, i):
        a = self.back_send_end_time_lst[i]
        return a

    def set_back_send_end_time_lst_elem(self, i, time):
        self.back_send_end_time_lst[i] = time

    def get_back_recv_start_time_lst_elem(self, i):
        a = self.back_recv_start_time_lst[i]
        return a

    def set_back_recv_start_time_lst_elem(self, i, time):
        self.back_recv_start_time_lst[i] = time

    def get_back_recv_end_time_lst_elem(self, i):
        a = self.back_recv_end_time_lst[i]
        return a

    def set_back_recv_end_time_lst_elem(self, i, time):
        self.back_recv_end_time_lst[i] = time

class PredTime:

    def __init__(self, num_procs: int,
                 num_split: int,
                 part_time_lst: list):

        self.num_procs=num_procs
        self.num_split=num_split
        self.part_time_lst=part_time_lst

    def __sync_for_send_recv(self, k, i):

        partk = self.part_time_lst[k]
        part_next = self.part_time_lst[k+1]

        send_end = partk.get_for_send_end_time_lst_elem(i)
        part_next.set_for_recv_end_time_lst_elem(i, send_end)

    def __pred_for_time_part(self, k: int, i: int):
        
        partk = self.part_time_lst[k]
        
        r_e_time = partk.get_for_recv_end_time_lst_elem(i)
        
        if i!=0:
            c_q_time = partk.get_for_compute_end_time_lst_elem(i-1)
            c_s_time = max(c_q_time, r_e_time)
        else:
            c_s_time = r_e_time

        partk.set_for_compute_start_time_lst_elem(i, c_s_time)

        c_e_time = c_s_time + partk.for_compute_time
        partk.set_for_compute_end_time_lst_elem(i, c_e_time)

        if i!=0:
            s_q_time = partk.get_for_send_end_time_lst_elem(i-1)
            s_s_time = max(s_q_time, c_e_time)
        else:
            s_s_time = c_e_time
        
        partk.set_for_send_start_time_lst_elem(i, s_s_time)

        s_e_time = s_s_time + partk.for_send_time
        partk.set_for_send_end_time_lst_elem(i, s_e_time)
        
        if k!=self.num_procs-1:
            self.__sync_for_send_recv(k, i)
        else:
            pass
             
    def pred_for_time(self):
        
        for i in range(self.num_split):
            for k in range(self.num_procs):
                self.__pred_for_time_part(k, i)

    def __sync_for_end_back_start(self):
        last = self.num_procs-1
        last_part = self.part_time_lst[last]
        
        last_split = self.num_split-1
        for_end_time = last_part.get_for_compute_end_time_lst_elem(last_split)
        last_part.set_back_compute_start_time_lst_elem(last_split,for_end_time)
                
    def __sync_back_send_recv(self, k, i):

        partk = self.part_time_lst[k]
        part_next = self.part_time_lst[k-1]

        send_end = partk.get_back_send_end_time_lst_elem(i)
        part_next.set_back_recv_end_time_lst_elem(i, send_end)

    def __pred_back_time_part(self, k: int, i: int):
        
        partk = self.part_time_lst[k]
        
        r_e_time = partk.get_back_recv_end_time_lst_elem(i)
        
        if i != self.num_split-1:
            c_q_time = partk.get_back_compute_end_time_lst_elem(i+1)
            c_s_time = max(c_q_time, r_e_time)
            
        elif k!=self.num_procs-1:
            c_s_time = r_e_time

        elif i == self.num_split-1:
            c_s_time = partk.get_for_compute_end_time_lst_elem(i)
        else:
            c_s_time = partk.get_back_compute_end_time_lst_elem(i+1)

        partk.set_back_compute_start_time_lst_elem(i, c_s_time)

        c_e_time = c_s_time + partk.back_compute_time
        partk.set_back_compute_end_time_lst_elem(i, c_e_time)

        if i != self.num_split-1:
            s_q_time = partk.get_back_send_end_time_lst_elem(i+1)
            s_s_time = max(s_q_time, c_e_time)
        else:
            s_s_time = c_e_time
        
        partk.set_back_send_start_time_lst_elem(i, s_s_time)

        s_e_time = s_s_time + partk.back_send_time
        partk.set_back_send_end_time_lst_elem(i, s_e_time)
        
        if k!=0:
            self.__sync_back_send_recv(k, i)
        else:
            pass
        
    def pred_back_time(self):

        self.__sync_for_end_back_start()
        
        for i in reversed(range(self.num_split)):
            for k in reversed(range(self.num_procs)):
                self.__pred_back_time_part(k, i)

    def print_time(self):

        for k in range(self.num_procs):
            partk = self.part_time_lst[k]
            
            c_end_time = partk.back_compute_end_time_lst[0]
            print(f"part {k} pred time: {c_end_time} s")
            print(partk.for_compute_end_time_lst)
            print(partk.back_compute_end_time_lst)

    def get_end_time(self):
        last_part=self.part_time_lst[-1]
        end_time = last_part.get_back_compute_end_time_lst_elem(-1)
        return end_time