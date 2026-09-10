"""
Patch flashinfer's JIT build system to remove the --compress-mode=size flag.
This flag is only supported on CUDA 13.x+ nvcc, but Modal's L4 instances
run CUDA 12.1.1 (injected by the container runtime from the host machine).

The flag is added by flashinfer's build system to compress CUDA fatbinaries.
Removing it makes the compiled kernels slightly larger but fully functional on CUDA 12.x.
"""
import os
import glob

fi_path = '/usr/local/lib/python3.10/site-packages/flashinfer'
patched_files = []
found_compress = []

print(f"Scanning flashinfer at: {fi_path}")

for fp in glob.glob(f'{fi_path}/**/*.py', recursive=True):
    try:
        with open(fp) as f:
            src = f.read()
    except Exception as e:
        continue

    if 'compress-mode' in src or 'compress_mode' in src:
        found_compress.append(fp)

print(f"Found {len(found_compress)} files with 'compress-mode' or 'compress_mode':")
for fp in found_compress:
    print(f"  {fp}")

# Patch by removing the unsupported flag in all its forms
for fp in found_compress:
    with open(fp) as f:
        src = f.read()
    
    new_src = src
    for old_flag in [
        "'--compress-mode=size'",
        '"--compress-mode=size"',
        "'--compress-mode=size',",
        '"--compress-mode=size",',
        "\"--compress-mode=size\"",
    ]:
        if old_flag in new_src:
            new_src = new_src.replace(old_flag, '""  # compress-mode removed: not supported on CUDA 12.x')
            print(f"  Replaced {old_flag!r} in {fp}")

    if new_src != src:
        with open(fp, 'w') as f:
            f.write(new_src)
        patched_files.append(fp)
        print(f"✓ Patched: {fp}")

if not patched_files:
    print("\nWARNING: No files patched. The flag may be dynamically generated.")
    print("Dumping all Python files containing 'compress':")
    for fp in glob.glob(f'{fi_path}/**/*.py', recursive=True):
        try:
            with open(fp) as f:
                src = f.read()
            if 'compress' in src.lower():
                for i, line in enumerate(src.splitlines(), 1):
                    if 'compress' in line.lower():
                        print(f"  {fp}:{i}: {line.strip()}")
        except:
            pass
else:
    print(f"\n✓ Successfully patched {len(patched_files)} file(s). JIT will now work on CUDA 12.x.")
