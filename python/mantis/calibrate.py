"""Microbenchmarks that calibrate the cost model (:mod:`mantis.cost`) on the target GPU.

Each figure is measured, not assumed (Volkov 2016: latency and occupancy effects on GPUs are not
predictable from specifications; CITATIONS.md). Results are cached per device under
``<cache>/calibration/``.

* ``op_latency_ns``: one dependent float FMA, a single warp (the critical-path cost).
* ``op_issue_ns``: one warp-instruction of independent FMAs at full occupancy, per SM (the
  throughput cost: an SM retires a warp-op every this many ns).
* ``sync_ns[k]``: one ``__syncthreads`` in a block of k warps, one block per SM.
* ``lat_ns[level]``: one dependent load (pointer chase) from shared memory, and from global memory
  with a working set that fits L1, L2, or neither (DRAM).
* ``access_ns[level]``: one warp-wide (coalesced) access of independent loads at full occupancy,
  per SM: the throughput cost, which is what independent straight-line accesses pay.
* ``dram_gbs``: streaming read bandwidth.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import subprocess

from . import device, harness

_SRC = r"""
#include <cstdint>
#include <cuda_runtime.h>

__global__ void k_lat(float* out, int n, float a)
{
    float x = threadIdx.x * 1e-3f;
#pragma unroll 64
    for (int i = 0; i < n; ++i) x = fmaf(x, a, 1e-7f);
    if (x == 12345.f) out[0] = x;
}

__global__ void k_issue(float* out, int n, float a)
{
    float x0 = threadIdx.x, x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3, x4 = x0 + 4, x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7;
#pragma unroll 8
    for (int i = 0; i < n; ++i)
    {
        x0 = fmaf(x0, a, 1e-7f); x1 = fmaf(x1, a, 1e-7f); x2 = fmaf(x2, a, 1e-7f); x3 = fmaf(x3, a, 1e-7f);
        x4 = fmaf(x4, a, 1e-7f); x5 = fmaf(x5, a, 1e-7f); x6 = fmaf(x6, a, 1e-7f); x7 = fmaf(x7, a, 1e-7f);
    }
    if (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 == 12345.f) out[0] = x0;
}

__global__ void k_sync(float* out, int n)
{
    __shared__ float s[1024];
    float x = threadIdx.x;
    for (int i = 0; i < n; ++i)
    {
        s[threadIdx.x] = x;
        __syncthreads();
        x += s[(threadIdx.x + 1) & (blockDim.x - 1)];
    }
    if (x == 12345.f) out[0] = x;
}

__global__ void k_chase(const uint32_t* __restrict__ next, uint32_t* out, int n)
{
    uint32_t j = 0;
    for (int i = 0; i < n; ++i) j = __ldcg(next + j);
    out[blockIdx.x] = j;
}

__global__ void k_chase_l1(const uint32_t* __restrict__ next, uint32_t* out, int n)
{
    uint32_t j = 0;
    for (int i = 0; i < n; ++i) j = __ldca(next + j);
    out[blockIdx.x] = j;
}

__global__ void k_chase_smem(const uint32_t* __restrict__ next, uint32_t* out, int n, int m)
{
    extern __shared__ uint32_t s[];
    for (int i = threadIdx.x; i < m; i += blockDim.x) s[i] = next[i];
    __syncthreads();
    if (threadIdx.x) return;
    uint32_t j = 0;
    for (int i = 0; i < n; ++i) j = s[j];
    out[blockIdx.x] = j;
}

template <bool CA>
__global__ void k_tput(const float* __restrict__ a, float* out, int n, unsigned mask)
{
    float acc = 0.f;
    unsigned i = (blockIdx.x * 977u + (threadIdx.x >> 5) * 131u) * 32u + (threadIdx.x & 31);
#pragma unroll 8
    for (int k = 0; k < n; ++k)
    {
        acc += CA ? __ldca(a + (i & mask)) : __ldcg(a + (i & mask));
        i += 32u * 61u;
    }
    if (acc == 12345.f) out[0] = acc;
}

__global__ void k_stream(const float4* __restrict__ a, float* out, size_t n)
{
    float acc = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    {
        const float4 v = a[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.f) out[0] = acc;
}

static float timed(void (*launch)(void*), void* arg, int reps)
{
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    launch(arg);
    cudaEventRecord(t0);
    for (int r = 0; r < reps; ++r) launch(arg);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return cudaGetLastError() == cudaSuccess ? ms / reps : -1.f;
}

struct Args { float* f; uint32_t* u; uint32_t* o; const void* p; int n, m, blocks, threads; size_t big; };
static void l_lat(void* a) { auto* x = (Args*)a; k_lat<<<1, 32>>>(x->f, x->n, 0.9999f); }
static void l_issue(void* a) { auto* x = (Args*)a; k_issue<<<x->blocks, x->threads>>>(x->f, x->n, 0.9999f); }
static void l_sync(void* a) { auto* x = (Args*)a; k_sync<<<x->blocks, x->threads>>>(x->f, x->n); }
static void l_chase(void* a) { auto* x = (Args*)a; k_chase<<<1, 1>>>((const uint32_t*)x->p, x->o, x->n); }
static void l_chase_l1(void* a) { auto* x = (Args*)a; k_chase_l1<<<1, 1>>>((const uint32_t*)x->p, x->o, x->n); }
static void l_smem(void* a)
{
    auto* x = (Args*)a;
    k_chase_smem<<<1, 128, x->m * sizeof(uint32_t)>>>((const uint32_t*)x->p, x->o, x->n, x->m);
}
static void l_tput_ca(void* a) { auto* x = (Args*)a; k_tput<true><<<x->blocks, x->threads>>>((const float*)x->p, x->f, x->n, (unsigned)x->m); }
static void l_tput_cg(void* a) { auto* x = (Args*)a; k_tput<false><<<x->blocks, x->threads>>>((const float*)x->p, x->f, x->n, (unsigned)x->m); }
static void l_stream(void* a) { auto* x = (Args*)a; k_stream<<<x->blocks, 256>>>((const float4*)x->p, x->f, x->big); }

// a random cyclic permutation of m words, stride-padded to one 128-byte line per hop
static uint32_t* chain(int m, int stride)
{
    uint32_t* h = new uint32_t[(size_t)m * stride]();
    uint32_t* order = new uint32_t[m];
    for (int i = 0; i < m; ++i) order[i] = i;
    uint64_t s = 88172645463325252ull;
    for (int i = m - 1; i > 0; --i)
    {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        const int j = (int)(s % (uint64_t)(i + 1));
        const uint32_t t = order[i]; order[i] = order[j]; order[j] = t;
    }
    for (int i = 0; i < m; ++i) h[(size_t)order[i] * stride] = order[(i + 1) % m] * stride;
    uint32_t* d;
    cudaMalloc(&d, (size_t)m * stride * sizeof(uint32_t));
    cudaMemcpy(d, h, (size_t)m * stride * sizeof(uint32_t), cudaMemcpyHostToDevice);
    delete[] h;
    delete[] order;
    return d;
}

extern "C" float lat(int n) { Args a{}; cudaMalloc(&a.f, 4); a.n = n; float t = timed(l_lat, &a, 5); cudaFree(a.f); return t; }
extern "C" float issue(int n, int blocks, int threads)
{
    Args a{}; cudaMalloc(&a.f, 4); a.n = n; a.blocks = blocks; a.threads = threads;
    float t = timed(l_issue, &a, 5); cudaFree(a.f); return t;
}
extern "C" float sync_(int n, int blocks, int threads)
{
    Args a{}; cudaMalloc(&a.f, 4); a.n = n; a.blocks = blocks; a.threads = threads;
    float t = timed(l_sync, &a, 5); cudaFree(a.f); return t;
}
// kind: 0 global through L2 (ld.cg), 1 global through L1 (ld.ca), 2 shared
extern "C" float chase(int kind, int bytes, int n)
{
    const int stride = kind == 2 ? 1 : 32, m = bytes / (4 * stride);
    Args a{}; a.n = n; a.m = m * stride;
    uint32_t* d = chain(m, stride);
    a.p = d;
    cudaMalloc(&a.o, 4 * 64);
    if (kind == 2 && a.m * 4 > 48 * 1024)
        cudaFuncSetAttribute(k_chase_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, a.m * 4);
    float t = timed(kind == 0 ? l_chase : kind == 1 ? l_chase_l1 : l_smem, &a, 3);
    cudaFree(d);
    cudaFree(a.o);
    return t;
}
// independent loads at full occupancy over a working set of `bytes` (a power of two): through L1
// (ld.ca) or L2 (ld.cg)
extern "C" float tput(int ca, size_t bytes, int n, int blocks, int threads)
{
    Args a{}; cudaMalloc(&a.f, 4); a.n = n; a.blocks = blocks; a.threads = threads; a.m = (int)(bytes / 4 - 1);
    void* p; cudaMalloc(&p, bytes); cudaMemset(p, 0, bytes); a.p = p;
    float t = timed(ca ? l_tput_ca : l_tput_cg, &a, 5);
    cudaFree(p); cudaFree(a.f);
    return t;
}
extern "C" float stream(size_t bytes, int blocks)
{
    Args a{}; cudaMalloc(&a.f, 4); a.big = bytes / 16; a.blocks = blocks;
    void* p; cudaMalloc(&p, bytes); cudaMemset(p, 0, bytes); a.p = p;
    float t = timed(l_stream, &a, 5);
    cudaFree(p); cudaFree(a.f);
    return t;
}
"""


@functools.cache
def _lib() -> ctypes.CDLL:
    out = harness.cache_dir() / "calibration" / f"bench_{hashlib.sha1(_SRC.encode()).hexdigest()[:12]}.so"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        cu = out.with_suffix(".cu")
        cu.write_text(_SRC)
        subprocess.run([harness.NVCC, "-O3", "-std=c++17", *harness.arch_flags(), "--shared",
                        "--compiler-options", "-fPIC", "-o", str(out), str(cu)], check=True,
                       capture_output=True, text=True)
    lib = ctypes.CDLL(str(out))
    for f, args in (("lat", [ctypes.c_int]), ("issue", [ctypes.c_int] * 3), ("sync_", [ctypes.c_int] * 3),
                    ("chase", [ctypes.c_int] * 3), ("stream", [ctypes.c_size_t, ctypes.c_int]),
                    ("tput", [ctypes.c_int, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int])):
        getattr(lib, f).restype = ctypes.c_float
        getattr(lib, f).argtypes = args
    return lib


def _diff(f, small: int, big: int) -> float:
    """ms per unit from two runs (removes the launch overhead): (f(big) - f(small)) / (big - small)."""
    return (min(f(big) for _ in range(3)) - min(f(small) for _ in range(3))) / (big - small)


def measure(dev: device.Device) -> dict:
    lib = _lib()
    full = dev.sms * max(dev.blocks_per_sm, 1)

    def warm():  # a light kernel alone leaves the clocks low; keep the GPU busy ~0.3 s first
        for _ in range(20):
            lib.issue(1 << 17, dev.sms * (dev.threads_per_sm // 256), 256)

    cal = {"device": dev.name, "cc": list(dev.cc)}
    warm()
    cal["op_latency_ns"] = _diff(lib.lat, 1 << 16, 1 << 20) * 1e6
    # 8 independent FMAs per iteration, 256-thread blocks filling every SM
    per_iter = _diff(lambda n: lib.issue(n, dev.sms * (dev.threads_per_sm // 256), 256), 1 << 12, 1 << 15)
    warp_ops_per_sm = 8 * dev.threads_per_sm // 32
    cal["op_issue_ns"] = per_iter * 1e6 / warp_ops_per_sm
    warm()
    cal["sync_ns"] = {str(k): _diff(lambda n, k=k: lib.sync_(n, dev.sms, 32 * k), 1 << 12, 1 << 15) * 1e6
                      for k in (1, 2, 4, 8, 16, 32)}
    warm()
    hops = (1 << 14, 1 << 17)
    l1 = 16 * 1024
    l2 = dev.l2 // 4 if dev.l2 else 1 << 20
    cal["lat_ns"] = {
        "shared": _diff(lambda n: lib.chase(2, 16 * 1024, n), *hops) * 1e6,
        "l1": _diff(lambda n: lib.chase(1, l1, n), *hops) * 1e6,
        "l2": _diff(lambda n: lib.chase(0, l2, n), *hops) * 1e6,
        "dram": _diff(lambda n: lib.chase(0, 256 << 20, n), *hops) * 1e6,
    }
    # ns per warp-access per SM at full occupancy (the throughput cost of independent accesses)
    warm()
    blocks, threads = dev.sms * (dev.threads_per_sm // 256), 256
    warps_per_sm = dev.threads_per_sm // 32

    def per_access(ca, nbytes):
        return _diff(lambda n: lib.tput(ca, nbytes, n, blocks, threads), 1 << 10, 1 << 13) * 1e6 / warps_per_sm

    cal["access_ns"] = {"l1": per_access(1, 16 * 1024), "l2": per_access(0, 1 << 21),
                        "dram": per_access(0, 1 << 30)}
    cal["access_ns"]["shared"] = cal["access_ns"]["l1"]  # same SRAM and port width
    nbytes = 1 << 30
    ms = min(lib.stream(nbytes, full) for _ in range(3))
    cal["dram_gbs"] = nbytes / (ms * 1e-3) / 1e9
    del full
    return cal


@functools.cache
def get(refresh: bool = False) -> dict:
    """The calibration of the target GPU, measured once and cached."""
    dev = device.target()
    key = hashlib.sha1(json.dumps([dev.name, dev.cc, harness.nvcc_version(), _SRC]).encode()).hexdigest()[:16]
    path = harness.cache_dir() / "calibration" / f"{key}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    cal = measure(dev)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal, indent=1))
    return cal
