# def parse_scheduler(scheduler, y=8):
#     def gg(x):
#         return x+y
#     fbconfig = []
#     if ";" in scheduler:
#         config_list = scheduler.split(";")
#     else:
#         config_list = [scheduler]
#     for config in config_list:
#         if not config or "," not in config:
#             continue
#         fbconfig.append([eval(n) for n in config.split(",")])
#     print(gg(1))
#     return fbconfig

# print(parse_scheduler("0.0,0.0,1.0,-1;0.0,0.0,1.0,-1"))

# def process_task(*args, **kwargs):
#     print(1, args)
#     print(2, kwargs)

# model = 14
# inputs = (16, 15, 14)
# kwargs = {
#     "model": model,
#     "inputs": inputs,
# }
# process_task(model=model, inputs=inputs)
from unittest.mock import patch 
class ProductionClass():
    def __init__(self):
        self.pp = 10
        
    def method(self, a, b, c):
        return a+b+c

def cucu(self, a, b, c):
    return a*b*c


thing = ProductionClass()
# new_cucu = cucu.__get__()
# with patch.object(thing, 'method', new_cucu):
#     print(thing.method(1, 1, 1))
    
# with patch.object(thing, 'pp', 5):
#     print(thing.pp)

# print(thing.method(1, 1, 1))
# print(thing.pp)

import contextlib

def get_patch(ob, threshold, nono, func=None):
    scheduler_step = 0
    
    def tre(self, a, b, c):
        nonlocal scheduler_step
        #scheduler_step_now = state["scheduler_step"]
        print((nono + threshold) <= scheduler_step, scheduler_step)
        use, ifchange = func(a+b+c)
        if ifchange:
            # state["scheduler_step"] = scheduler_step_now + 1
            scheduler_step = scheduler_step + 1
        print(use)
        
    new_tre = tre.__get__(ob)
    
    @contextlib.contextmanager
    def patch_orig():
        with patch.object(ob, 'method', new_tre):
            yield
    
    # def get_scheduler_step(new_scheduler_step):
    #     state["scheduler_step"] = new_scheduler_step
    
    # patch_orig.step = get_scheduler_step
    return patch_orig


def test():
    
    def change_scheduler(use):
        return use, True
    
    patch_func = get_patch(
        ob=thing,
        threshold=1, 
        nono=1,
        func=change_scheduler
    )
    
    with patch_func():
        thing.method(1, 2, 3)
        thing.method(4, 5, 6)
        thing.method(7, 8, 9)

def test2():
    step = 1
    if step:
        print("1")
    else:
        print("2")
        
#test()
test2()