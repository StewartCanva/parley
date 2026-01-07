// Copyright 2025 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Font selection strategies for customizing font fallback behavior.

use crate::analysis::AnalysisDataSources;
use crate::analysis::cluster::CharCluster;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;
use linebender_resource_handle::FontData;

/// Represents the result of font selection for a text cluster.
#[derive(Debug, Clone)]
pub enum FontSelectionResult {
    /// Use the specified font for the entire cluster
    UseFont(crate::shape::SelectedFont),
    /// Use a fallback font for the entire cluster
    UseFallbackSegment(FallbackSegment),
    /// No font available - skip this cluster
    NoFont,
}

/// A fallback font selection for a single cluster.
///
/// This represents a font to use for an entire grapheme cluster when primary fonts fail.
/// The `char_range` must match the character range of the cluster being processed - it
/// can span multiple Unicode codepoints (e.g., base character + combining marks).
#[derive(Debug, Clone)]
pub struct FallbackSegment {
    /// Character range within the text (not byte range).
    /// Must match the cluster's character range exactly.
    pub char_range: Range<usize>,
    /// Font to use for the entire cluster
    pub font: FontData,
    /// Font synthesis settings (bold/italic emulation)
    pub synthesis: fontique::Synthesis,
}

impl FallbackSegment {
    /// Create a new fallback segment with explicit synthesis.
    ///
    /// The `char_range` should match the character range of the cluster being shaped.
    pub fn new(char_range: Range<usize>, font: FontData, synthesis: fontique::Synthesis) -> Self {
        Self {
            char_range,
            font,
            synthesis,
        }
    }
}

/// Strategy trait for customizing font selection behavior.
///
/// Implementors can define custom logic for selecting fonts when primary fonts
/// fail to provide glyphs for specific characters.
///
/// # Statefulness
///
/// Strategies should be designed to be stateless during font selection, or at least
/// safe to call multiple times with different text. The same strategy instance will
/// be reused across multiple layout operations on the same `FontContext`.
pub trait FontSelectionStrategy: Send + Sync {
    /// Determine the fallback mode for system fonts.
    fn fallback_mode(&self) -> crate::shape::FallbackMode;

    /// Select a font for a specific text cluster.
    ///
    /// This method is called for each text cluster during shaping. It should:
    /// 1. Try primary fonts using the provided `FontSelector`
    /// 2. If primary fonts fail, apply custom fallback logic
    /// 3. Return appropriate `FontSelectionResult`
    ///
    /// # Important: Cluster-Based Selection
    ///
    /// This method operates on **grapheme clusters**, which are indivisible units for text shaping.
    /// A cluster may contain multiple Unicode codepoints (e.g., base character + combining marks).
    ///
    /// When returning `UseFallbackSegment`, you must select a **single font** that can handle
    /// the **entire cluster**. You cannot split a cluster across multiple fonts, as this would
    /// break text shaping (ligatures, contextual forms, etc.).
    ///
    /// The `char_range` parameter indicates the character range of the current cluster.
    /// If returning `UseFallbackSegment`, the segment's `char_range` must match this range exactly.
    #[allow(private_interfaces)]
    fn select_font_for_cluster<'a, 'b>(
        &self,
        cluster: &mut CharCluster,
        font_selector: &mut crate::shape::FontSelector<'a, 'b>,
        text: &str,
        char_range: Range<usize>,
        font_weight: crate::FontWeight,
        font_style: crate::FontStyle,
        analysis_data_sources: &AnalysisDataSources,
    ) -> FontSelectionResult;
}

/// Default font selection strategy that preserves original Parley behavior.
///
/// This strategy:
/// - Uses system fallbacks when primary fonts fail
/// - Maintains all existing performance characteristics
/// - Provides identical behavior to pre-strategy Parley
#[derive(Clone, Default)]
pub struct DefaultFontSelectionStrategy;

impl DefaultFontSelectionStrategy {
    /// Create a new default font selection strategy.
    pub fn new() -> Self {
        Self
    }
}

impl FontSelectionStrategy for DefaultFontSelectionStrategy {
    fn fallback_mode(&self) -> crate::shape::FallbackMode {
        // System fallbacks enabled (preserves original behavior)
        crate::shape::FallbackMode::WithSystemFallback
    }

    #[allow(private_interfaces)]
    fn select_font_for_cluster<'a, 'b>(
        &self,
        char_cluster: &mut CharCluster,
        font_selector: &mut crate::shape::FontSelector<'a, 'b>,
        _text: &str,
        _char_range: Range<usize>,
        _font_weight: crate::FontWeight,
        _font_style: crate::FontStyle,
        analysis_data_sources: &AnalysisDataSources,
    ) -> FontSelectionResult {
        // Pure delegation to original FontSelector (preserves all performance)
        if let Some(selected_font) = font_selector.select_font(char_cluster, analysis_data_sources)
        {
            FontSelectionResult::UseFont(selected_font)
        } else {
            FontSelectionResult::NoFont
        }
    }
}

/// Canva-specific font selection strategy.
///
/// This strategy implements Canva's requirements:
/// - Try primary fonts first
/// - If primary fonts fail, check Unicode range mappings
/// - If no mapping exists, return `NoFont` (no system fallback)
/// - Prevents system fallbacks by using `PrimaryFontsOnly` mode
#[derive(Clone, Default)]
pub struct CanvaFontSelectionStrategy {
    ranges: Vec<UnicodeRangeEntry>,
}

#[derive(Debug, Clone)]
struct UnicodeRangeEntry {
    range: Range<u32>,
    font: FontData,
    weight: crate::FontWeight,
    style: crate::FontStyle,
    synthesis: fontique::Synthesis,
}

impl CanvaFontSelectionStrategy {
    /// Create a new Canva font selection strategy.
    pub fn new() -> Self {
        Self { ranges: Vec::new() }
    }

    /// Add a Unicode range mapping with synthesis.
    ///
    /// When primary fonts fail, characters in this range will use the specified font
    /// with the specified synthesis (bold/italic emulation).
    pub fn add_unicode_range_with_synthesis(
        &mut self,
        range: Range<u32>,
        font: FontData,
        weight: crate::FontWeight,
        style: crate::FontStyle,
        synthesis: fontique::Synthesis,
    ) {
        self.ranges.push(UnicodeRangeEntry {
            range,
            font,
            weight,
            style,
            synthesis,
        });
    }

    /// Get the number of configured Unicode ranges.
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Check if there are no configured Unicode ranges.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}

impl FontSelectionStrategy for CanvaFontSelectionStrategy {
    fn fallback_mode(&self) -> crate::shape::FallbackMode {
        // System fallbacks prevented by using FontContext::with_system_fonts(false)
        crate::shape::FallbackMode::PrimaryFontsOnly
    }

    #[allow(private_interfaces)]
    fn select_font_for_cluster<'a, 'b>(
        &self,
        cluster: &mut CharCluster,
        font_selector: &mut crate::shape::FontSelector<'a, 'b>,
        text: &str,
        char_range: Range<usize>,
        font_weight: crate::FontWeight,
        font_style: crate::FontStyle,
        analysis_data_sources: &AnalysisDataSources,
    ) -> FontSelectionResult {
        // TODO(conor) When Canva*Strategy is extracted, reuse identical logic in font_selector
        // Adapted from https://github.com/Canva/canva/blob/ccf7be5a5d103725e5dfe8d404d70d3fe2bcc262/web/src/services/ripple/document/interpreters/fonts/font_loader.ts#L1011-L1020
        fn get_distance(from_weight: u16, from_italics: bool, target_weight: u16, target_style: crate::FontStyle) -> i32 {
            // The sum of the weights has lowest priority, it biases towards 'Normal' (< 2000).
            let sum = distance_to_normal(from_weight) + distance_to_normal(target_weight);
            // The next highest priority is the distance between weights (< 1000).
            let diff = (from_weight as i32 - target_weight as i32).abs();
            // The highest priority is whether it's italics.
            // Add penalty when italic styles don't match
            let italic = if from_italics != (target_style == crate::FontStyle::Italic) {
                1 // Penalty for italic mismatch
            } else {
                0 // No penalty when italics match
            };
            // Return the prioritised results as a single distance.
            sum + 2000 * (diff + 1000 * italic)
        }

        // Adjust a weight so that Normal is 0, and all expected values are unique.
        fn distance_to_normal(weight: u16) -> i32 {
            (weight as i32 - 400).abs() + if weight > 400 { 1 } else { 0 }
        }

        // Step 1: Try primary fonts only (no system fallback)
        if let Some(selected_font) = font_selector.select_font(cluster, analysis_data_sources) {
            return FontSelectionResult::UseFont(selected_font);
        }

        // Step 2: Primary fonts failed, try Unicode ranges
        if self.ranges.is_empty() {
            return FontSelectionResult::NoFont;
        }

        // Convert char range to byte range once for efficiency
        let char_ranges = vec![char_range.clone()];
        let byte_ranges = crate::shape::char_ranges_to_byte_ranges(text, &char_ranges);
        let byte_range = &byte_ranges[0];
        let cluster_text = &text[byte_range.clone()];

        let target_weight = font_weight.value() as u16;
        let target_italics = font_style == crate::FontStyle::Italic;

        // Find the font that can handle ALL characters AND has the closest style/weight match
        // We need to pick a single font because clusters are indivisible shaping units
        let mut closest_font: Option<(&UnicodeRangeEntry, i32)> = None;

        for entry in &self.ranges {
            // Check if this font can handle all characters in the cluster
            let mut can_handle_all = true;

            for ch in cluster_text.chars() {
                let char_code = ch as u32;
                if !entry.range.contains(&char_code) {
                    can_handle_all = false;
                    break;
                }
            }

            if !can_handle_all {
                continue;
            }

            // This font can handle all characters, compute its distance
            let distance = get_distance(
                target_weight,
                target_italics,
                entry.weight.value() as u16,
                entry.style,
            );

            // Update closest_font if this is closer (or it's the first match)
            match closest_font {
                None => {
                    closest_font = Some((entry, distance));
                }
                Some((_, current_distance)) => {
                    if distance < current_distance {
                        closest_font = Some((entry, distance));
                    }
                }
            }
        }

        match closest_font {
            Some((entry, _)) => FontSelectionResult::UseFallbackSegment(FallbackSegment::new(
                char_range,
                entry.font.clone(),
                entry.synthesis,
            )),
            None => {
                // No single font can handle all characters in this cluster
                FontSelectionResult::NoFont
            }
        }
    }
}
