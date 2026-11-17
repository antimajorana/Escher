mod build;
mod canvas_ops;
mod markers;
mod panel_left;
mod panel_right;
mod popover;
mod state;

#[path = "../combine_json.rs"]
mod json;

#[path = "../combine_kinematics.rs"]
mod kinematics;

#[cfg(test)]
use std::time::Duration;
#[cfg(test)]
use papertilling_core::defect::surgery::DefectSpec;
#[cfg(test)]
use crate::backend::AppBackend;
#[cfg(test)]
use crate::canvas2d::{Canvas2D, Scene2D};
#[cfg(test)]
use crate::live_combine::spec_world_pos;

#[cfg(test)]
#[path = "../combine_tab_tests.rs"]
mod tests;

pub use kinematics::MotionMode;
pub(crate) use state::{MANUAL_R_MAX, MANUAL_R_MIN};
pub use state::{CombineTabState, ToolMode};
