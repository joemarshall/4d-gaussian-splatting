import torch
import os


""" A tensor which is backed by a file on disk, and only loads a filtered selection of the file into device memory at a time"""
class FileBackedTensor:
    def __init__(self,fname,device,dtype=torch.float32):
        super().__init__()
        self.fname =fname
        self.dtype = dtype
        try:
            filesize = os.stat(fname).st_size
            # first 8 int32s are the dimensions
            self._attach_tensor((filesize-8*4)//dtype.itemsize)

            element_size = 1
            element_shape=[]
            for x in self.dimension_tensor:
                if x == 0:
                    break
                element_size*=x.item()
                element_shape.append(x)
            self.element_shape = element_shape
            self.mmap_tensor = self.mmap_tensor.reshape(-1,*element_shape)            
        except Exception as e:
            self._attach_tensor(0)
            print(e) 
        
        self.loaded_filter = None
        self.loaded_shape = None
        self.device = device

    def _attach_tensor(self, data_len,shape = None):
        self.map_storage = torch.UntypedStorage.from_file(self.fname,nbytes=data_len*self.dtype.itemsize + 8*4,shared=True)
        self.dimension_tensor = torch.tensor(self.map_storage,dtype=torch.int32,device="cpu")[0:8]
        start_offset = (8*4) // self.dtype.itemsize
        self.mmap_tensor = torch.tensor(self.map_storage,dtype=self.dtype)[start_offset:]
        if shape is not None:
            self.mmap_tensor = self.mmap_tensor.reshape(-1,*shape)
        else:
            new_shape = []
            for x in self.dimension_tensor:
                if x == 0:
                    break
                new_shape.append(x)
            self.mmap_tensor = self.mmap_tensor.reshape(-1,*new_shape)
        self.dimension_tensor[0:len(self.mmap_tensor.shape)-1] = torch.tensor(self.mmap_tensor.shape[1:],dtype=torch.int32)

    def load_to_device(self, filter: torch.Tensor):
        # filter is a boolean tensor of the same length as the first dimension of the mmap_tensor
        if self.loaded_filter is not None and torch.equal(filter, self.loaded_filter):
            return self.loaded_tensor
        else:
            loaded_tensor = self.mmap_tensor[filter].to(self.device)
            self.loaded_filter = filter
            self.loaded_shape = loaded_tensor.shape
            return loaded_tensor
        
    # restore from device tensor to file, and if shape has changed update the file length by copying data around
    def save_from_device(self, loaded_tensor: torch.Tensor):
        if self.loaded_shape==None:
            # just add to end of mmap tensor
            if len(self.mmap_tensor.shape)==0 or self.mmap_tensor.nelement()==0:
                print("Empty tensor, writing")
                self._attach_tensor(loaded_tensor.nelement(),shape=loaded_tensor.shape[1:])
                self.mmap_tensor[:] = loaded_tensor.cpu()
            else:
                assert(self.mmap_tensor.shape[1:] == loaded_tensor.shape[1:])
                old_count = self.mmap_tensor.shape[0]
                new_num_elements = self.mmap_tensor.nelement() + loaded_tensor.nelement()
                print("Appending tensor, new els:",new_num_elements)
                self._attach_tensor(new_num_elements)
                print(self.mmap_tensor.shape)
                self.mmap_tensor[old_count:] = loaded_tensor.cpu()
        elif loaded_tensor.shape == self.loaded_shape:
            self.mmap_tensor[self.loaded_filter] = loaded_tensor.cpu()
        else:
            size_diff = loaded_tensor.shape[0] - self.loaded_shape[0]
            if size_diff < 0:
                # removing some entries
                new_size = self.mmap_tensor.shape[0] + size_diff
                old_shape = self.mmap_tensor.shape
                new_shape = (new_size,) + self.mmap_tensor.shape[1:]
                new_num_elements = 1
                for x in new_shape:
                    new_num_elements *= x

                # copy: new elements plus some from end
                self.mmap_tensor[self.loaded_filter] = torch.cat((loaded_tensor.cpu(),self.mmap_tensor[size_diff:])).squeeze()
                # then resize file by reloading the tensor again
                self._attach_tensor(new_num_elements)
            elif size_diff>0:
                # adding some entries
                # resize file and add them at end
                new_size = self.mmap_tensor.shape[0] + size_diff
                old_shape = self.mmap_tensor.shape
                new_shape = (new_size,) + self.mmap_tensor.shape[1:]
                new_num_elements = 1
                for x in new_shape:
                    new_num_elements *= x

                # copy old elements to existing places
                self.mmap_tensor[self.loaded_filter] = loaded_tensor[0:self.loaded_shape[0]].cpu()
                self._attach_tensor(new_num_elements)
                # add new elements at end
                self.mmap_tensor[old_shape[0]:] = loaded_tensor[self.loaded_shape[0]:].cpu()

    
class TimeAndScaleChunkedTensor:
    """ Store multiple tensors as chunks based on time and scale and only load them to cuda when we need them."""
    def __init__(self,device,min_chunk_size,max_chunk_size = 1.0):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.max_log_chunk_size = torch.log2(torch.tensor(max_chunk_size/min_chunk_size)).ceil()

        self.chunks = {}
        self.tracked_tensors = ["_times","_scales"]
        self.device = device

    def add_tracked_tensor(self,name):
        self.tracked_tensors.append(name)

    def add_values(self, values, chunk_id):
        # add values to chunk with this ID, creating if needed
        chunk_id = tuple(chunk_id.tolist())
        if chunk_id not in self.chunks:
            self.chunks[chunk_id] = {}
            for name in self.tracked_tensors:
                self.chunks[chunk_id][name] = values[name].detach().to("cpu")
        else:
            # adding to existing chunk only happens if a gaussian has changed time scale or moved in time
            # so hopefully shouldn't happen too often...
            for name in self.tracked_tensors:
                self.chunks[chunk_id][name] = torch.cat((self.chunks[chunk_id][name],values[name].detach().to("cpu")),dim=0)

    def get_chunk_boundaries(self,chunk_id):
        chunk_level,chunk_time_idx = chunk_id
        level_size = self.min_chunk_size * (2**chunk_level)
        start_time = level_size * (chunk_time_idx - 1)
        end_time = start_time + level_size*3
        return start_time,end_time


    def move_chunks_to_device(self, min_time,max_time):
        # combine all chunks for given time range and load to device
        # all chunks we used are marked dirty at this point,
        # if we call update_chunks we will recreate all copied chunks with 
        # updated time/scale values
        chunks_to_combine = []
        chunks_to_clear = []
        for chunk_level,chunk_time_idx in self.chunks.keys():
            start_time,end_time = self.get_chunk_boundaries((chunk_level,chunk_time_idx))
            if start_time <= min_time and end_time >= max_time:
                chunks_to_combine.append(self.chunks[(chunk_level,chunk_time_idx)])
                chunks_to_clear.append((chunk_level,chunk_time_idx))
        for chunk_id in chunks_to_clear:
            del self.chunks[chunk_id]
        if len(chunks_to_combine)==0:
            print("No chunks to combine for time range",min_time,max_time)
            return None,None,None
        # move everything across to device as combined tensor
        ret_chunks={}
        for name in self.tracked_tensors:
            name_tensors = [x[name] for x in chunks_to_combine]
            if len(name_tensors)==0:
                print("Empty chunk:",name)
                print(chunks_to_combine)
            ret_chunks[name]=torch.cat(name_tensors,dim=0).to(self.device)
        times,scales = ret_chunks["_times"],ret_chunks["_scales"]
        del ret_chunks["_times"]
        del ret_chunks["_scales"]    
#        print("After moving to device num chunks:",len(self.chunks))
        return ret_chunks,times,scales

    def update_chunks_from_device(self,device_data,times,scales):
        # add all these values to chunks, creating if needed
        point_chunk_ids = self.get_chunk_ids_for_time_scale(times,scales)
        chunk_ids = torch.unique(point_chunk_ids,dim=0)
        device_data["_times"]= times
        device_data["_scales"]=scales
#        print("Got n unique chunks:",len(chunk_ids))
        for chunk_id in chunk_ids:
            # print(chunk_id)
            chunk_filter = torch.all(point_chunk_ids==chunk_id,dim=1)
            # print(chunk_filter)
            all_chunk_vals = {}
            for name in self.tracked_tensors:
                if name not in device_data:
                    raise ValueError(f"Device data missing tracked tensor {name}")         
                name_data = device_data[name]
                chunk_values = name_data[chunk_filter]
                all_chunk_vals[name]= chunk_values
            self.add_values(all_chunk_vals,chunk_id)
#        print("After update num chunks:",len(self.chunks))

    def get_chunk_ids_for_time_scale(self,time,scale):
        # get chunk id for given time and scale
        chunk_levels = (torch.log2(scale/self.min_chunk_size).floor())
        # Limit chunk levels to a maximum value
        chunk_levels = torch.clamp(chunk_levels, min= 0 , max=self.max_log_chunk_size)
        level_size = (self.min_chunk_size * (2**chunk_levels))
        time_offsets = (time / level_size).floor()
        chunk_ids = torch.hstack((chunk_levels.unsqueeze(1), time_offsets.unsqueeze(1)))
        return chunk_ids

def test_file_backed_tensor():
    f=FileBackedTensor("test.tmp",device="cuda",dtype=torch.float32)
    f.save_from_device(torch.tensor([[1],[2],[3]]))
    print(f.mmap_tensor)
    filter_selection = torch.cat((torch.tensor([True,False,True]),torch.zeros(f.mmap_tensor.shape[0]-3,dtype=torch.bool)))
    tensor = f.load_to_device(filter_selection)
    print(tensor)
    # save, same length, should just overwrite positions 0 and 2
    f.save_from_device(torch.tensor([[4],[5]],dtype=tensor.dtype,device=tensor.device))

    print("After write 4 and 5 to positions 0 and 2:")
    print(f.mmap_tensor)

    filter_selection = torch.zeros_like(f.mmap_tensor,dtype=torch.bool)
    filter_selection[-2] = True

    tensor = f.load_to_device(filter_selection)

    # save, longer length
    f.save_from_device(torch.tensor([[8],[9],[10],[11]],dtype=tensor.dtype,device=tensor.device))
    print("After replacing position -2 with 8, 9,10,11")
    print(f.mmap_tensor)


    filter_selection = torch.zeros_like(f.mmap_tensor,dtype=torch.bool)
    filter_selection[0] = True
    filter_selection[3] = True
    filter_selection[4] = True

    tensor = f.load_to_device(filter_selection)
    f.save_from_device(torch.tensor([[99],[12]],dtype=tensor.dtype,device=tensor.device))
    print("After replacing positions 0 and 3 with 99 and 12, and removing position 4")
    print(f.mmap_tensor)



def test_chunked_tensors():
    chunked_tensor = TimeAndScaleChunkedTensor(device="cuda",min_chunk_size=.1)
    chunked_tensor.add_tracked_tensor("test1")
    chunked_tensor.add_tracked_tensor("test2")

    def make_values(names):
        values ={}
        for x in names:
            values[x] = torch.rand(100)
        scales = torch.rand(100)*100.0 + 0.1
        times= torch.rand(100)*100.0
        return values,scales,times
    


    values,scales,times = make_values(["test1","test2"])
    chunked_tensor.update_chunks_from_device(values,times,scales)

    for x in sorted(chunked_tensor.chunks.keys()):
        print(x,chunked_tensor.chunks[x]["test1"].shape)

    device_data,device_times,device_scales = chunked_tensor.move_chunks_to_device(min_time=5,max_time=6)
    print(device_data)
    chunked_tensor.update_chunks_from_device(device_data,device_times,device_scales)

    for x in sorted(chunked_tensor.chunks.keys()):
        print(x,chunked_tensor.chunks[x]["test1"].shape)

if __name__=="__main__":
    test_chunked_tensors()