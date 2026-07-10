//! Unit tests for the tag system in wgsl-ping-pong-pipeline

use wgsl_ping_pong_pipeline::pipeline::{Stage, StageConfig, PipelineStage};
use wgsl_ping_pong_pipeline::wgpu_utils::ComputeContext;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::CommandEncoder;
use anyhow::Result;

/// A test tag struct with an id field
#[derive(Debug, Clone, PartialEq)]
struct TestTag {
    id: u64,
}

impl TestTag {
    fn new(id: u64) -> Self {
        Self { id }
    }
}

// A simple custom stage for testing
struct TestCustomStage {
    name: String,
    vector_dim: usize,
    batch_size: usize,
}

impl PipelineStage for TestCustomStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn vector_dim(&self) -> usize {
        self.vector_dim
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn encode(
        &self,
        _encoder: &mut CommandEncoder,
        _input: &wgpu::Buffer,
        _output: &wgpu::Buffer,
        _side_inputs: &HashMap<String, Arc<wgpu::Buffer>>,
    ) -> Result<()> {
        Ok(())
    }

    fn initialize(&mut self, _context: &ComputeContext) -> Result<()> {
        Ok(())
    }

    fn requires_initialization(&self) -> bool {
        false
    }
}

impl std::fmt::Debug for TestCustomStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TestCustomStage")
            .field("name", &self.name)
            .field("vector_dim", &self.vector_dim)
            .field("batch_size", &self.batch_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that StageConfig can store and forward tags correctly
    #[test]
    fn test_stage_config_tag_storage() {
        let stage = Stage::new("test_stage", "@compute @workgroup_size(1) fn main() {}", 2, 1024);
        
        // Create StageConfig with a tag
        let mut config = StageConfig::<TestTag>::Standard {
            stage,
            tag: Some(TestTag::new(1)),
        };
        
        // Verify tag is stored
        assert_eq!(config.tag(), &Some(TestTag::new(1)));
    }

    /// Test that forward_tag exchanges tags correctly
    #[test]
    fn test_stage_config_forward_tag_standard() {
        let stage = Stage::new("test_stage", "@compute @workgroup_size(1) fn main() {}", 2, 1024);
        let mut config = StageConfig::<TestTag>::Standard {
            stage,
            tag: Some(TestTag::new(42)),
        };
        
        // Forward a new tag, should get the old one back
        let old_tag = config.forward_tag(Some(TestTag::new(100)));
        assert_eq!(old_tag, Some(TestTag::new(42)));
        assert_eq!(config.tag(), &Some(TestTag::new(100)));
    }

    /// Test forward_tag with None
    #[test]
    fn test_stage_config_forward_tag_with_none() {
        let stage = Stage::new("test_stage", "@compute @workgroup_size(1) fn main() {}", 2, 1024);
        let mut config = StageConfig::<TestTag>::Standard {
            stage,
            tag: None,
        };
        
        // Forward a tag to a config with None, should get None back
        let old_tag = config.forward_tag(Some(TestTag::new(1)));
        assert_eq!(old_tag, None);
        assert_eq!(config.tag(), &Some(TestTag::new(1)));
    }

    /// Test forward_tag on custom stage
    #[test]
    fn test_stage_config_forward_tag_custom() {
        let custom_stage = Box::new(TestCustomStage {
            name: "custom".to_string(),
            vector_dim: 2,
            batch_size: 1024,
        });
        let mut config = StageConfig::<TestTag>::Custom {
            stage: custom_stage,
            tag: Some(TestTag::new(99)),
        };
        
        // Forward a new tag
        let old_tag = config.forward_tag(Some(TestTag::new(200)));
        assert_eq!(old_tag, Some(TestTag::new(99)));
        assert_eq!(config.tag(), &Some(TestTag::new(200)));
    }

    /// Test that tags can be any struct (in this case, a struct with id)
    #[test]
    fn test_tag_as_struct_with_id() {
        let stage = Stage::new("test", "@compute @workgroup_size(1) fn main() {}", 2, 8);
        let config = StageConfig::<TestTag>::Standard {
            stage,
            tag: Some(TestTag::new(123)),
        };
        
        // Verify we can store and retrieve the struct with id
        assert_eq!(config.tag().as_ref().unwrap().id, 123);
    }
    
    #[test]
    fn test_tag_forwarding_with_id() {
        let stage = Stage::new("test", "@compute @workgroup_size(1) fn main() {}", 2, 8);
        let mut config = StageConfig::<TestTag>::Standard {
            stage,
            tag: Some(TestTag::new(123)),
        };
        
        // Forward a new tag
        let old_tag = config.forward_tag(Some(TestTag::new(456)));
        assert_eq!(old_tag.unwrap().id, 123);
        assert_eq!(config.tag().as_ref().unwrap().id, 456);
    }
}
