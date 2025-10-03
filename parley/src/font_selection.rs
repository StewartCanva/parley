// Copyright 2025 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Font selection strategies for customizing font fallback behavior.

use crate::Font;
use std::ops::Range;

/// Represents the result of font selection for a text cluster.
#[derive(Debug, Clone)]
pub enum FontSelectionResult {
    /// Use the specified font for the entire cluster
    UseFont(crate::shape::SelectedFont),
    /// Use pre-segmented fallback fonts - each segment has its own font
    UseFallbackSegments(Vec<FallbackSegment>),
    /// No font available - skip this cluster
    NoFont,
}

/// A pre-segmented text range with its assigned font.
#[derive(Debug, Clone)]
pub struct FallbackSegment {
    /// Character range within the text (not byte range)
    pub char_range: Range<usize>,
    /// Font to use for this character range
    pub font: Font,
    /// Font synthesis settings (bold/italic emulation)
    pub synthesis: fontique::Synthesis,
}

impl FallbackSegment {

    /// Create a new fallback segment with explicit synthesis.
    pub fn new(char_range: Range<usize>, font: Font, synthesis: fontique::Synthesis) -> Self {
        Self { char_range, font, synthesis }
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
/// be reused across multiple layout operations on the same FontContext.
pub trait FontSelectionStrategy: Send + Sync {
    /// Determine the fallback mode for system fonts.
    fn fallback_mode(&self) -> crate::shape::FallbackMode;


    /// Select a font for a specific text cluster.
    ///
    /// This method is called for each text cluster during shaping. It should:
    /// 1. Try primary fonts using the provided FontSelector
    /// 2. If primary fonts fail, apply custom fallback logic
    /// 3. Return appropriate FontSelectionResult
    ///
    /// # CRITICAL: UseFallbackSegments Limitations
    ///
    /// **UseFallbackSegments can only be used for single-cluster scenarios.**
    ///
    /// The segments **must** cover exactly the character range of the current
    /// cluster being processed. The shaping engine advances by exactly one cluster
    /// after processing segments, regardless of how many characters the segments cover.
    ///
    /// **Multi-cluster fallback is NOT supported via UseFallbackSegments** - it will
    /// cause text positioning corruption. For complex fallback scenarios spanning
    /// multiple clusters, implement the logic in `select_font_for_cluster` to return
    /// `UseFont` or `NoFont` for each individual cluster as it's processed.
    ///
    /// This limitation exists because cluster boundaries and character boundaries
    /// don't always align (due to grapheme clusters, ligatures, etc.).
    #[allow(private_interfaces)]
    fn select_font_for_cluster<'a, 'b>(
        &self,
        cluster: &mut swash::text::cluster::CharCluster,
        font_selector: &mut crate::shape::FontSelector<'a, 'b>,
        text: &str,
        char_range: Range<usize>,
    ) -> FontSelectionResult;
}

/// Default font selection strategy that preserves original Parley behavior.
///
/// This strategy:
/// - Uses system fallbacks when primary fonts fail
/// - Maintains all existing performance characteristics
/// - Provides identical behavior to pre-strategy Parley
#[derive(Clone)]
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
        cluster: &mut swash::text::cluster::CharCluster,
        font_selector: &mut crate::shape::FontSelector<'a, 'b>,
        _text: &str,
        _char_range: Range<usize>,
    ) -> FontSelectionResult {
        // Pure delegation to original FontSelector (preserves all performance)
        if let Some(selected_font) = font_selector.select_font(cluster) {
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
/// - If primary fonts fail, check unicode range mappings
/// - If no mapping exists, return NoFont (no system fallback)
/// - Prevents system fallbacks by using PrimaryFontsOnly mode
#[derive(Clone)]
pub struct CanvaFontSelectionStrategy {
    ranges: Vec<UnicodeRangeEntry>,
}

#[derive(Debug, Clone)]
struct UnicodeRangeEntry {
    range: Range<u32>,
    font: Font,
    synthesis: fontique::Synthesis,
}

impl CanvaFontSelectionStrategy {
    /// Create a new Canva font selection strategy.
    pub fn new() -> Self {
        Self {
            ranges: Vec::new(),
        }
    }

    /// Add a unicode range mapping with synthesis.
    ///
    /// When primary fonts fail, characters in this range will use the specified font
    /// with the specified synthesis (bold/italic emulation).
    pub fn add_unicode_range_with_synthesis(&mut self, range: Range<u32>, font: Font, synthesis: fontique::Synthesis) {
        self.ranges.push(UnicodeRangeEntry { range, font, synthesis });
    }

    /// Get the number of configured unicode ranges.
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Check if there are no configured unicode ranges.
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
        cluster: &mut swash::text::cluster::CharCluster,
        font_selector: &mut crate::shape::FontSelector<'a, 'b>,
        text: &str,
        char_range: Range<usize>,
    ) -> FontSelectionResult {
        // Step 1: Try primary fonts only (no system fallback) - clean single method call
        if let Some(selected_font) = font_selector.select_font(cluster) {
            return FontSelectionResult::UseFont(selected_font);
        }

        // Step 2: Primary fonts failed, try unicode ranges (custom Canva logic - unchanged)
        if self.ranges.is_empty() {
            return FontSelectionResult::NoFont;
        }

        // Fix O(n×m) performance regression: convert char range to byte range once
        // instead of calling text.chars().skip() which re-iterates from start every time
        let char_ranges = vec![char_range.clone()];
        let byte_ranges = crate::shape::char_ranges_to_byte_ranges(text, &char_ranges);
        let byte_range = &byte_ranges[0];

        // Work with the actual text slice for this character range
        let cluster_text = &text[byte_range.clone()];
        let mut segments = Vec::new();

        // Now we can iterate efficiently over just the cluster's characters
        for (local_index, ch) in cluster_text.chars().enumerate() {
            let char_code = ch as u32;
            let mut found_match = false;

            for entry in &self.ranges {
                if entry.range.contains(&char_code) {
                    let absolute_char_position = char_range.start + local_index;
                    segments.push(FallbackSegment::new(
                        absolute_char_position..(absolute_char_position + 1),
                        entry.font.clone(),
                        entry.synthesis.clone()
                    ));
                    found_match = true;
                    break;
                }
            }

            if !found_match {
                // At least one character can't be handled - return NoFont for entire cluster
                // This preserves text positioning and prevents missing glyphs
                return FontSelectionResult::NoFont;
            }
        }

        FontSelectionResult::UseFallbackSegments(segments)
    }
}