Here is the fully corrected, production-ready Product Requirements Document (PRD). This version reflects a **True Pipeline Cascade** designed for maximum parallel throughput, where each stage owns its own isolated pair of Ping-Pong buffers.
# **High-Throughput Pipelined Compute Cascade (wgpu) — Specification**
## **1. Stage Composition & Topology**
* **Linear Chain**: Support an arbitrary number of discrete compute shader stages chained sequentially: \text{Stage}_0 \rightarrow \text{Stage}_1 \rightarrow \dots \rightarrow \text{Stage}_{N-1}.
* **Pipeline Capacity**: The pipeline must support N concurrently active batches of data across N stages without artificial caps.
* **Shader Agnosticism**: Each stage accepts any valid compute shader module operating on a flat data array. The framework is not responsible for validating inner shader logic, only its data boundary safety.
## **2. Isolated Double-Buffered Architecture**
* **Per-Stage Private Pools**: Each individual stage i owns exactly **2 private buffers** (\text{Buffer } A_i and \text{Buffer } B_i). Stages do *not* share a global memory pool.
* **Complete Memory Isolation**: There is zero memory overlap between what \text{Stage}_i is writing and what \text{Stage}_{i+1} is reading on any given tick.
* **Maximum Throughput Concurrency**: Because of complete memory isolation, all N stages must execute their compute dispatches simultaneously during a single tick, allowing the GPU to run all shaders fully concurrently.
## **3. Data Flow & Tick-Based Propagation**
* **Synchronous Ticking**: The entire pipeline advances via an explicit clock step (.tick()) which submits all stage dispatches to the GPU command queue at once.
* **N-Tick Propagation**: A single batch of data takes exactly N ticks to propagate completely through N stages.
* **Local Role Alternation**: On every tick, each stage flips its internal read/write targets (Ping \rightarrow Pong) using an internal boolean toggle.
* **Inter-Stage Hand-off**: Stage i+1 reads its input directly from the active *output* buffer of Stage i from the preceding tick. Data never moves physically between stages during normal ticks—only bind group references alternate.
* **IO Interface**: The user can write a new input batch to \text{Stage}_0 on any given tick, and read processed output from \text{Stage}_{N-1} when available.
## **4. Batch Processing & Vector Support**
* **Uniform Dimension**: The pipeline enforces a fixed vector dimension (2\text{D}, 3\text{D}, 4\text{D}, \dots, M\text{D}) configured at initialization. All elements across all batches must share this dimension.
* **Variable Batch Size**: A "batch" consists of an array of these vectors. While vector dimensions are immutable, the *number of vectors* (batch size) can fluctuate over time.
## **5. Global Dynamic Resize Mechanism**
* **Simultaneous Allocation**: When a batch size change is requested, all 2 \times N buffers across all stages are reallocated to the new size in a single operation.
* **Data Migration & Preservation**: Before the old buffers are dropped, the framework must issue 2 \times N discrete copy_buffer_to_buffer commands on the GPU. This transfers all in-flight data from every stage's old inner buffers to their newly resized counterparts.
* **Atomic Transition**: The pipeline stalls for exactly one host-side clock tick during a resize to allow the GPU to copy the data and safely re-bake all stage bind groups. Normal operation resumes immediately after.
* **Hardware-Sized Slicing**:
* If a resize expands the pipeline, existing smaller batches traveling through the system are naturally padded with zeros up to the new buffer boundaries.
* If a resize shrinks the pipeline, data boundaries truncate cleanly based on the new dimension limits.
## **6. GPU Memory Management**
* **Strict VRAM Residency**: All intermediate data blocks across all stages must remain strictly within GPU memory space.
* **No CPU Roundtrips**: No data may be piped back to host CPU RAM for layout manipulation, size tracking, or buffering during normal pipeline execution or resizing.
## **7. Validation & Error Handling**
* **Size Validation**: The framework must validate that incoming data dimensions match the configured vector structure before allowing entry into \text{Stage}_0.
* **Graceful Lifecycle Failures**: If any buffer reallocation fails during a resize (e.g., VRAM exhaustion), the system must halt gracefully, leaving existing in-flight data intact on the old buffers for recovery or logging.
