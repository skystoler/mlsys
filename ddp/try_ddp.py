import torch
import os
import logging
import time
import pickle
import gc
from typing import Union, List, Optional, Tuple, Any
import contextlib
import socket
import threading
import traceback
from multiprocessing.reduction import ForkingPickler

import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel.scatter_gather import is_namedtuple
from torch.nn.parallel._functions import Scatter, Gather
from torch.nn.parallel import DistributedDataParallel as DDP

torch.classes.load_library('v4789cc_trt_executor_extension_py38.so')
    
logger = logging.getLogger()


class ExceptionOutput(Exception):
    pass


class MPDistRunner:
    def __init__(self, *, rank=None, persist_attrs=None, temp_attrs=None):
        self.rank = rank
        self.persist_attrs = persist_attrs or {}
        self.temp_attrs = temp_attrs or {}

        self.array_size = None
        self.shared_array = None
        self.input_queue = None
        self.output_queue = None
        self.status = None
        self.lock = None

        self.processes = []

    def __del__(self):
        self.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

    @property
    def world_size(self):
        return torch.cuda.device_count()

    @property
    def start_method(self):
        return "spawn"

    @property
    def max_input_size(self):
        return 1 << 30

    def clone(self, *, rank=None):
        return self.__class__(rank=rank, persist_attrs={**self.persist_attrs})

    def init_env(self, *, master_addr=None, master_port=None):
        if master_addr is None:
            if "MASTER_ADDR" not in os.environ:
                master_addr = "127.0.0.1"
                os.environ["MASTER_ADDR"] = master_addr
        else:
            os.environ["MASTER_ADDR"] = master_addr
        if master_port is None:
            if "MASTER_PORT" not in os.environ:
                master_port = find_free_port()
                os.environ["MASTER_PORT"] = str(master_port)
        else:
            os.environ["MASTER_PORT"] = str(master_port)

    def start(self, args=(), kwargs=None, *, timeout=None):
        if self.processes:
            raise RuntimeError("Processes are already started")

        if kwargs is None:
            kwargs = {}

        self.init_env()

        world_size = self.world_size
        if world_size <= 0:
            raise ValueError(f"World size {world_size} is invalid")

        start_method = self.start_method
        mp_ = mp.get_context(start_method)

        barrier = mp_.Barrier(world_size)
        array_size = mp_.Value("q", 0, lock=False)
        shared_array = mp_.Array("c", self.max_input_size, lock=False)
        input_queue = mp_.JoinableQueue(maxsize=1)
        output_queue = mp_.Queue(maxsize=1)
        exception_queues = [mp_.Queue(maxsize=1) for _ in range(world_size)]
        status = mp_.Value("i", 0, lock=True)
        processes = []

        try:
            for rank in range(world_size):
                process = mp_.Process(
                    target=self.worker,
                    args=(
                        self.clone(rank=rank),
                        args,
                        kwargs,
                        barrier,
                        array_size,
                        shared_array,
                        input_queue if rank == 0 else None,
                        output_queue if rank == 0 else None,
                        exception_queues[rank] if rank == 0 else None,
                        status if rank == 0 else None,
                    ),
                    daemon=True,
                )
                process.start()
                processes.append(process)

            output_queue.get(timeout=timeout)
            exceptions = []
            for rank, exception_queue in enumerate(exception_queues):
                if not exception_queue.empty():
                    exceptions.append((rank, exception_queue.get()))
            if exceptions:
                msg = "\n".join(f"Rank {rank}: {exception}" for rank, exception in exceptions)
                raise RuntimeError(f"Exceptions occurred:\n{msg}")
        except Exception:
            self.terminate()
            raise

        self.array_size = array_size
        self.shared_array = shared_array
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.exception_queues = exception_queues
        self.status = status
        self.lock = threading.RLock()
        self.processes = processes

        return self

    def terminate(self):
        self.array_size = None
        self.shared_array = None
        self.input_queue = None
        self.output_queue = None
        self.exception_queues = None
        self.status = None

        for process in self.processes:
            if process.is_alive():
                process.terminate()
        self.processes = []

    def restart(self, *args, **kwargs):
        self.terminate()
        self.start(*args, **kwargs)

    def is_alive(self):
        return self.processes and all(process.is_alive() for process in self.processes)

    def is_idle(self):
        if not self.is_alive():
            return False

        with self.status.get_lock():
            return self.status.value == 0

    def is_almost_idle(self):
        if not self.is_alive():
            return False

        with self.status.get_lock():
            return self.status.value in (0, 2)

    def __call__(self, args=(), kwargs=None, *, timeout=None):
        if kwargs is None:
            kwargs = {}

        with self.lock:
            if not self.processes:
                raise RuntimeError("Processes are not started")
            for rank, process in enumerate(self.processes):
                if not process.is_alive():
                    raise RuntimeError(f"Process {rank} is not alive")

            data = ForkingPickler.dumps((args, kwargs), pickle.HIGHEST_PROTOCOL)
            data_size = len(data)
            if data_size > self.max_input_size:
                raise RuntimeError(f"Data size {data_size} exceeds maximum size {self.max_input_size}")
            self.array_size.value = data_size
            self.shared_array[:data_size] = data
            if timeout is not None:
                begin_time = time.time()
            self.input_queue.put(True, timeout=timeout)
            self.input_queue.join()
            if timeout is not None:
                end_time = time.time()
                duration = end_time - begin_time
                timeout = max(0, timeout - duration)
            output = self.output_queue.get(timeout=timeout)
            
            # should gather here?
            exceptions = []
            for rank, exception_queue in enumerate(self.exception_queues):
                if (rank == 0 and isinstance(output, ExceptionOutput)) or not exception_queue.empty():
                    exceptions.append((rank, exception_queue.get()))
            if exceptions:
                msg = "\n".join(f"Rank {rank}: {exception}" for rank, exception in exceptions)
                raise RuntimeError(f"Exceptions occurred:\n{msg}")
            return output

    def init_process_group(self):
        if dist.is_initialized():
            return
        dist.init_process_group(world_size=self.world_size, rank=self.rank)

    def destroy_process_group(self):
        if not dist.is_initialized():
            return
        dist.destroy_process_group()

    def init_processor(self, *args, **kwargs):
        module = kwargs['model']
        if not isinstance(module, torch.nn.Module):
            self.module = torch.jit.load(module, map_location=f'cuda:{self.rank}')
        else:
            self.module = module.to(f'cuda:{self.rank}')

    def process_task(self, *args, **kwargs):
        inputs = kwargs['inputs']
        return self.module(inputs[self.rank])
        
    @classmethod
    def worker(
        cls,
        runner,
        args,
        kwargs,
        barrier,
        array_size,
        shared_array,
        input_queue,
        output_queue,
        exception_queue,
        status,
    ):
        runner.init_process_group()

        try:
            try:
                runner.init_processor(*args, **kwargs)
            except Exception as e:
                exception = RuntimeError(f"Failed to initialize processor: {e}\n{traceback.format_exc()}")
                if exception_queue is not None:
                    exception_queue.put(exception)
            barrier.wait()
            if output_queue is not None:
                output_queue.put(True)

            while True:
                if input_queue is None:
                    barrier.wait()
                else:
                    input_queue.get()
                    barrier.wait()

                if status is not None:
                    with status.get_lock():
                        status.value = 1

                data_size = array_size.value
                value_bytes = shared_array[:data_size]

                output = None
                exception = None

                try:
                    input_args, input_kwargs = ForkingPickler.loads(value_bytes)
                except Exception as e:
                    exception = RuntimeError(f"Failed to unpickle data: {e}\n{traceback.format_exc()}")

                if output_queue is not None:
                    while not output_queue.empty():
                        output_queue.get()
                if exception_queue is not None:
                    while not exception_queue.empty():
                        exception_queue.get()
                if input_queue is not None:
                    input_queue.task_done()

                if exception is None:
                    try:
                        output = runner.process_task(*input_args, **input_kwargs)
                    except Exception as e:
                        exception = RuntimeError(f"Failed to process task: {e}\n{traceback.format_exc()}")

                if status is not None:
                    with status.get_lock():
                        status.value = 2

                if exception_queue is not None:
                    if exception is not None:
                        exception_queue.put(exception)

                if output_queue is not None:
                    if exception is None:
                        output_queue.put(output)
                    else:
                        output_queue.put(ExceptionOutput())

                barrier.wait()

                if status is not None:
                    with status.get_lock():
                        status.value = 0
        finally:
            runner.destroy_process_group()


def find_free_port():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    

def load_input_from_pickle_to_device(pickle_path, device):
    images_t, masks_t, token_ids_t = simple_load_input_from_pickle(pickle_path)
    images_t = images_t.to(device)
    masks_t = masks_t.to(device)
    token_ids_t = token_ids_t.to(device)
    return images_t, masks_t, token_ids_t

def simple_load_input_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as pickle_file:
        data_dict = pickle.load(pickle_file)
    images, masks, token_ids = data_dict['stacked_images_t'], data_dict['stacked_masks_t'], data_dict['stacked_token_ids_t']
    images_t = images.half()
    masks_t = masks.half()
    token_ids_t = token_ids.to(torch.int64)
    return images_t, masks_t, token_ids_t


def replicate_model_and_inputs_by_dist(module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], rank: int, world_size: int) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    """
    Replicate the model and inputs to the specified device.
    Returns:
        Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]: The replicated model and inputs.
    """

    device = torch.device(f'cuda:{rank}')
    replicated_model = module.to(device)
    
    #batch_num, device_id = _find_batch_size_and_device_id(inputs, dim=0)
    if isinstance(inputs, tuple):
        need_replicate_inputs = inputs[0]
    
    need_replicate_inputs = need_replicate_inputs.to(device)
    batch_num = need_replicate_inputs.shape[0]
    
    if batch_num == 1:
        return module, inputs
    
    chunk_batch_size = 1 if batch_num < world_size else batch_num // world_size
    
    replicate_input_size = [chunk_batch_size] + list(need_replicate_inputs.shape[1:])
    replicated_inputs = torch.zeros(replicate_input_size, device=device)
    
    scatter_list = [
        need_replicate_inputs[i * chunk_batch_size: (i + 1) * chunk_batch_size] 
        for i in range(world_size - 1)
    ]
    scatter_list.append(
        need_replicate_inputs[
            batch_num - (world_size - 1) * chunk_batch_size:
        ]
    )
    if rank != 0:
        scatter_list = None
    
    dist.scatter(tensor=replicated_inputs, scatter_list=scatter_list)
    inputs = replicated_inputs + inputs[1:]
    replicated_inputs = tuple(inp.to(device) for inp in inputs)
    return replicated_model, (replicated_inputs, inputs[1:])


def setup(rank: int, world_size: int):
    """
    Initialize the process group for distributed inference.
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:29500',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)
    torch.manual_seed(42)


def cleanup():
    """
    Clean up the distributed process group.
    """
    dist.destroy_process_group()


def scatter(inputs, target_devices, dim=0):
    r"""Slice tensors into approximately equal chunks and distributes them across given GPUs.

    Duplicates references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_devices, None, dim, obj)
        if is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(scatter_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
        return [obj for _ in target_devices]

    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None  # type: ignore[assignment]
    return res


def gather(outputs: Any, target_device: Union[int, torch.device], dim: int = 0) -> Any:
    r"""Gather tensors from different GPUs on a specified device."""

    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all(len(out) == len(d) for d in outputs):
                raise ValueError("All dicts must have the same number of keys")
            return type(out)((k, gather_map([d[k] for d in outputs])) for k in out)
        if is_namedtuple(out):
            return type(out)._make(map(gather_map, zip(*outputs)))
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        res = gather_map(outputs)
    finally:
        gather_map = None  # type: ignore[assignment]
    return res


def run_inference(rank: int, module: str, inputs: Tuple[torch.Tensor, ...], world_size: int, outputs_list: List):
    """
    The function each spawned process will execute to perform inference.

    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
        module (torch.nn.Module): The model to be used for inference.
        inputs (Tuple[torch.Tensor, ...]): The inputs for inference.
        outputs_list (List): A shared list to collect outputs from each process.
    """
    try:
        setup(rank, world_size)
        
        # loaded again when necessary
        if not isinstance(module, torch.nn.Module):
            module = torch.jit.load(module, map_location=f'cuda:{rank}')
        else:
            module = module.to(f'cuda:{rank}')
        
        # Wrap the model with DDP
        ddp_model = DDP(module, device_ids=[rank])
        ddp_model.eval()

        with torch.no_grad():
            output = ddp_model(*(inputs[rank]))
        
        outputs_list[rank] = output
        
    except Exception as e:
        print(f"Exception in process {rank}: {e}")
    finally:
        del module
        del ddp_model
        cleanup()
        gc.collect()


class DDPInferencer:
    def __init__(
        self, 
        model,
        world_size: Optional[int] = None
    ):
        self.model = model
        
        self.world_size = torch.cuda.device_count() if world_size is None else world_size
        if self.world_size < 1:
            raise ValueError("No GPUs available for inference.")
        
        # Ensure world_size does not exceed available GPUs
        self.world_size = min(self.world_size, torch.cuda.device_count())
        
        # Manager list to collect outputs from each process
        self.manager = mp.Manager()
        self.outputs = self.manager.list([None] * self.world_size)
        
    def replicate(self, inputs):
        """
        Return a list of replicated inputs
        """
        target_devices = [i for i in range(torch.cuda.device_count())]
        return scatter(inputs, target_devices=target_devices, dim=0)
    
    def gather(self, outputs):
        if len(outputs) == 1:
            return outputs[0]
        return gather(outputs, target_device=torch.device('cpu'), dim=0)
    
    def cat_outputs(self, outputs):
        ddp_outputs = []
        for output in zip(*outputs):
            if isinstance(output[0], torch.Tensor):
                ddp_outputs.append(torch.cat(output, dim=0))
            else:
                # ddp_outputs.append([torch.cat(t, dim=0) for t in zip(*output[0])])
                ddp_outputs.append([self.cat_outputs(t) for t in zip(*output)])
        return ddp_outputs
          
    # def warp_modules(self):
    #     modules = self.modules
    #     def wrap_getstate(self):
    #         print("wrap getstate")
    #         return
        
    #     def wrap_setstate(self, state):
    #         print("wrap setstate")
        
    #     for module in modules:
    #         setattr(module, '__getstate__', wrap_getstate)
    #         setattr(module, '__setstate__', wrap_setstate)
        
    #     return modules
        
    def infer(self, inputs) -> List[torch.Tensor]:
        """
        Spawn processes for each GPU and perform inference.

        Returns:
            List[torch.Tensor]: List of outputs from each GPU.
        """
        if inputs is None:
            raise ValueError("Inputs are not set.")
        
        print("Process spawn begin.")
        mp.spawn(
            run_inference,
            args=(self.model, inputs, self.world_size, self.outputs),
            nprocs=self.world_size,
            join=True
        )
        print("Gathering outputs begin.")
        print(self.outputs[0].device)
        # dist.all_gather(self.outputs, )
        outputs = [o.to(torch.device("cuda:0")) for o in self.outputs]
        ddp_outputs = self.gather(outputs)
        print("Gathering outputs done.")

        if isinstance(ddp_outputs, torch.Tensor):
            return ddp_outputs
        return tuple(ddp_outputs)
    

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1024, 1024)
    
    def forward(self, x):
        return self.linear(x)


def test_simple_model(model=SimpleModel()):
    inputs = torch.randn(8, 1024, 1024)
    test_base(model, inputs)

 
"""
import pickle
data ={
    'stacked_images_t': stacked_images_t,
    'stacked_masks_t': stacked_masks_t,
    'stacked_token_ids_t': stacked_token_ids_t,
}
with open('text_recog_input.pkl', 'wb') as file:
    pickle.dump(data, file)
"""
# "uniocr_wfeat_241024152028_uniocr_add_spzonly_241023_85.pth"
def test_text_recog_trt_model(model="vc0c279_prt_ft13_emptyrecog_invert_0411_147_opt.ts"):
    inputs = simple_load_input_from_pickle("text_recog_input.pkl")
    print(inputs[0].shape, inputs[1].shape, inputs[2].shape)
    inputs = [input[:8] for input in inputs]
    print(inputs[0].shape, inputs[1].shape, inputs[2].shape)
    # inputs = (
    #     torch.randn([42, 3, 64, 832]).half(), 
    #     torch.randn([42, 64, 832]).half(), 
    #     torch.randn([42, 1]).to(torch.int64),
    # )
    test_base(model, *inputs)
    

def test_card_model(model="v266b7a_IDFRTAX_ckpt-97-cls_acc_99.26_tr.pth", input_device=torch.device('cuda:0')):
    inputs = (
        torch.randn([1, 3, 448, 448]).to(torch.float32).to(input_device), 
        torch.randn([1, 2048]).to(torch.int64).to(input_device),
        torch.randn([1, 2048, 4]).to(torch.int64).to(input_device), 
        torch.randn([1, 2048]).to(torch.int64).to(input_device), 
        torch.randn([1, 2048]).to(torch.int32).to(input_device),
    )
    test_base(model, *inputs, dtype=torch.float32)

# from ocr.utils.predict_ops import load_script
# import underframe.torch as uf_torch
def test_base(model, *inputs):
    # if not isinstance(model, torch.nn.Module):
    #     models = [torch.jit.load(model, map_location=device) for device in devices]
        # tec = uf_torch.context.TorchExecutionContext(device=device, dtype=dtype)
        # with tec.omni_context():
        #     model = load_script(model, freeze=True, optimize=True)
        # model = uf_torch.nn.BatchLimitedModule(model, batch_limit=32)
    # if not isinstance(model, torch.nn.Module):
    #     model = torch.jit.load(model, map_location=torch.device('cuda:0'))
    # start = time.time()
    # outputs = model(*inputs)
    # end = time.time()
    # infer_time = end - start
    # print("Infer Time:", infer_time)
    
    print("Initialize DDP Inferencer begin.")
    inferencer = DDPInferencer(model)
    print("Initialize DDP Inferencer done.")
    
    # Perform distributed inference
    start = time.time()
    print("Replicating inputs begin.")
    replicate_inputs = inferencer.replicate(inputs)
    print("Replicating inputs done.")
    ddp_output = inferencer.infer(replicate_inputs)
    end = time.time()
    ddp_infer_time = end - start
    print("DDP Infer Time:", ddp_infer_time)
    
    # torch.testing.assert_close(ddp_output, outputs)

def test_runner():
    inputs = simple_load_input_from_pickle("text_recog_input.pkl")
    print(inputs[0].shape, inputs[1].shape, inputs[2].shape)
    inputs = [input[:8] for input in inputs]
    print(inputs[0].shape, inputs[1].shape, inputs[2].shape)
    
    mp_runner = MPDistRunner()
    mp_runner.start(model="vc0c279_prt_ft13_emptyrecog_invert_0411_147_opt.ts")
    target_devices = [i for i in range(torch.cuda.device_count())]
    replicated_inputs = scatter(inputs, target_devices=target_devices, dim=0)
    outputs = mp_runner(inputs=replicated_inputs)
 
if __name__ == "__main__":
    # test_simple_model()
    # test_card_model()
    # test_text_recog_trt_model()
    test_runner()