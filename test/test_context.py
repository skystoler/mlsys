import contextlib

@contextlib.contextmanager
def patch_current_context(new_context):
    global current_context
    old_context = current_context
    current_context = new_context
    try:
        yield
    finally:
        current_context = old_context

current_context = 0
new_ctx = 1

@patch_current_context(new_ctx)
def test_patch_current_context():
    print(current_context)
    
print(current_context)
test_patch_current_context()
print(current_context)