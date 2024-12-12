import torch
import os
import logging
import time
from typing import Union, List, Optional, Tuple, Any, Dict, Sequence

import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from torch.nn.parallel.scatter_gather import is_namedtuple
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel import DistributedDataParallel as DDP

torch.classes.load_library('v4789cc_trt_executor_extension_py38.so')
    
logger = logging.getLogger()


def run_inference(rank: int, module, inputs: Tuple[torch.Tensor, ...], world_size: int, outputs_list: List):
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
        # replicated_module, replicated_inputs = replicate_model_and_inputs_by_dist(module, inputs, rank, world_size)
        replicated_module, replicated_inputs = replicate_model_and_inputs_by_dp(module, inputs, rank, world_size)
        
        # Wrap the model with DDP
        ddp_model = DDP(replicated_module, device_ids=[rank])
        ddp_model.eval()
        
        with torch.no_grad():
            output = ddp_model(*replicated_inputs)
        
        # Move output to CPU and append to the shared list
        if isinstance(output, torch.Tensor):
            outputs_list[rank] = [output.cpu()]
        elif isinstance(output, (list, tuple)):
            outputs_list[rank] = [o.cpu() for o in output]
    
    except Exception as e:
        print(f"Exception in process {rank}: {e}")
    finally:
        cleanup()


def setup(rank: int, world_size: int):
    """
    Initialize the process group for distributed inference.

    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
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


def replicate_model_and_inputs_by_dist(module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], rank: int, world_size: int) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    """
    Replicate the model and inputs to the specified device.

    Args:
        device (torch.device): The target device.

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


# need do outside of run inference
def replicate_model_and_inputs_by_dp(module, inputs: Tuple[torch.Tensor, ...], rank: int, world_size: int) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    device = torch.device(f'cuda:{rank}')
    replicated_model = module.to(device) if isinstance(module, torch.nn.Module) else torch.jit.load(module, map_location=device)
    target_gpus = [i for i in range(world_size)]
    scatter_inputs_list, scatter_kwargs_list = scatter_kwargs(inputs, kwargs=None, target_gpus=target_gpus)
    replicated_inputs = tuple(input.to(device) for input in scatter_inputs_list[rank])
    return replicated_model, replicated_inputs


def scatter_kwargs(
    inputs: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]],
    target_gpus: Sequence[Union[int, torch.device]],
    dim: int = 0,
) -> Tuple[Tuple[Any, ...], Tuple[Dict[str, Any], ...]]:
    r"""Scatter with support for kwargs dictionary."""
    scattered_inputs = scatter(inputs, target_gpus, dim)
    scattered_kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(scattered_inputs) < len(scattered_kwargs):
        scattered_inputs.extend(
            () for _ in range(len(scattered_kwargs) - len(scattered_inputs))
        )
    elif len(scattered_kwargs) < len(inputs):
        scattered_kwargs.extend(
            {} for _ in range(len(scattered_inputs) - len(scattered_kwargs))
        )
    return scattered_inputs, scattered_kwargs


def scatter(inputs, target_gpus, dim=0):
    r"""Slice tensors into approximately equal chunks and distributes them across given GPUs.

    Duplicates references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(scatter_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
        return [obj for _ in target_gpus]

    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None  # type: ignore[assignment]
    return res


class DDPInferencer:
    def __init__(
        self, 
        module, 
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        world_size: Optional[int] = None
    ):
        self.module = module
        self.inputs = inputs
        
        # Validate inputs are tensors
        for inp in self.inputs:
            if not isinstance(inp, torch.Tensor):
                raise ValueError("All inputs must be torch.Tensor instances.")
        
        self.world_size = torch.cuda.device_count() if world_size is None else world_size
        if self.world_size < 1:
            raise ValueError("No GPUs available for inference.")
        
        # Ensure world_size does not exceed available GPUs
        self.world_size = min(self.world_size, torch.cuda.device_count())
        
        # Manager list to collect outputs from each process
        self.manager = mp.Manager()
        self.outputs = self.manager.list([None] * self.world_size)

    def infer(self) -> List[torch.Tensor]:
        """
        Spawn processes for each GPU and perform inference.

        Returns:
            List[torch.Tensor]: List of outputs from each GPU.
        """
        mp.spawn(
            run_inference,
            args=(self.module, self.inputs, self.world_size, self.outputs),
            nprocs=self.world_size,
            join=True
        )
        ddp_outputs = tuple(torch.cat(output, dim=0) for output in zip(*self.outputs))
        return ddp_outputs[0] if len(ddp_outputs) == 1 else ddp_outputs
     
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1024, 1024)
    
    def forward(self, x):
        return self.linear(x)


def test_simple_model():
    model = SimpleModel()
    inputs = torch.randn(8, 1024, 1024)
    test_base(model, inputs)
    del model


def test_vpf_model(model_name):
    inputs = torch.randn([42, 3, 64, 832]).half(), torch.randn([42, 64, 832]).half(), torch.randn([42, 1]).to(torch.int64)
    test_base(model_name, *inputs)
    del model


def test_base(model, *inputs):
    # Initialize the DDPInferencer
    inferencer = DDPInferencer(module=model, inputs=inputs)
    
    # Perform distributed inference
    start = time.time()
    ddp_output = inferencer.infer()
    torch.cuda.synchronize()
    end = time.time()
    ddp_infer_time = end - start
    print("Combined Output Shape:", [output.shape for output in ddp_output] if isinstance(ddp_output, tuple) else ddp_output.shape, "DDP Infer Time:", ddp_infer_time)
    
    if not isinstance(model, torch.nn.Module):
        model = torch.jit.load(model, map_location=torch.device('cuda:0'))
    start = time.time()
    outputs = model(*inputs)
    end = time.time()
    infer_time = end - start
    print("Output Shape:", [output.shape for output in outputs] if isinstance(outputs, tuple) else outputs.shape, "Infer Time:", infer_time)
    
    torch.testing.assert_close(ddp_output, outputs)

 
if __name__ == "__main__":
    test_simple_model()
    # test_vpf_model(model_name="vc0c279_prt_ft13_emptyrecog_invert_0411_147_opt.ts")