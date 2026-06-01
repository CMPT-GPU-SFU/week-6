# Week 6: Tensor Cores & MMA Programming

This repository contains the hands-on mini-project for **Week 6 of the Directed Reading: GPU Programming for Machine Learning Acceleration** course. The focus of this lab is understanding hardware matrix-multiply units (Tensor Cores) and programming them using warp-level primitives.

## Lab Objectives
* Implement a high-performance Half-Precision GEMM (HGEMM) kernel using CUDA `wmma` intrinsics.
* Extend the tiling structure to compare performance differences when utilizing `TF32` instructions.
* Conduct a precision and performance analysis against a production baseline (`cuBLAS`) utilizing a large transformer-scale linear layer ($4096 \times 4096 \times 11008$).

## Repository Structure
This repository utilizes a two-branch architecture to serve as a turnkey educational resource:
* **`main` / `template` Branch:** The student starter kit. Contains the complete CMake build system, host-side memory orchestration, cuBLAS benchmarking harness, and automated validation scripts. Core hardware-level kernel logic is left as a implementation exercise.
* **`solution` Branch:** The reference implementation. Features fully verified, compiling WMMA implementations with active validation, timing metrics, and error analysis code.

## Prerequisites & Hardware
* **Hardware:** Compute Capability 7.0+ (Volta, Ampere, Ada Lovelace, or Hopper architecture).
* **Software:** Toolchain supporting CUDA 11.0+ and CMake 3.18+.
