# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-25

### Added

- Initial release of wgsl-ping-pong-pipeline
- Core `Pipeline` type with ping-pong buffering pattern
- `Stage` struct for standard compute stages with WGSL shaders
- `PipelineStage` trait for custom stage implementations
- `PipelineBuilder` for ergonomic pipeline construction
- `VariableSizePipeline` for pipelines with per-stage buffer sizes
- `ComputeContext` wrapper for wgpu Device and Queue management
- Support for side inputs (buffers bound to stages but not in data flow)
- Tag-based data tracking through pipeline stages
- Buffer resize support (both batch size and vector dimension)
- Comprehensive test suite with 30 tests
- Builder example demonstrating basic usage

### Features

- **Ping-Pong Buffering**: Each stage has isolated input/output buffers that alternate, allowing data to flow continuously through the pipeline
- **Generic Design**: Works with any WGSL compute shader
- **Custom Stages**: Full support for custom `PipelineStage` implementations via trait system
- **Variable-Size Pipelines**: Stages can have independent buffer sizes for efficient memory usage
- **Thread-Safe**: All types are thread-safe with Arc-based resource management

[Unreleased]: https://github.com/larsjoost/wgsl-ping-pong-pipeline/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/larsjoost/wgsl-ping-pong-pipeline/releases/tag/v0.1.0
