import os
import time
import torch

GB = 10  # 想占用的显存大小（GiB）
DEVICE = "cuda:0"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA 不可用，无法占用 GPU 显存。")

device = torch.device(DEVICE)
torch.cuda.set_device(device)

# float32 每个元素 4 字节
num_elements = int(GB * 1024**3 / 4)

print(f"[START] PID={os.getpid()} on {DEVICE}, target={GB} GiB", flush=True)

x = torch.empty(num_elements, dtype=torch.float32, device=device)
x.fill_(1.0)  # 真正触发显存分配

torch.cuda.synchronize()
allocated = torch.cuda.memory_allocated(device) / 1024**3
reserved = torch.cuda.memory_reserved(device) / 1024**3

print(f"[READY] allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB", flush=True)
print("[HOLD] 正在持续占用显存，按 Ctrl+C 或 kill 结束。", flush=True)

while True:
    time.sleep(3600)