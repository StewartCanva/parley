// Copyright 2021 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Text shaping implementation using `harfrust`for shaping
//! and `swash` for text analysis.

use core::ops::RangeInclusive;

use alloc::vec::Vec;
use std::sync::Arc;

use super::layout::Layout;
use super::resolve::{RangedStyle, ResolveContext, Resolved};
use super::style::{Brush, FontFeature, FontVariation};
use crate::inline_box::InlineBox;
use crate::lru_cache::LruCache;
use crate::util::nearly_eq;
use crate::{FontData, swash_convert};

/// Controls whether system fallback fonts are used when primary fonts fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackMode {
    /// Use system fallback fonts if primary fonts can't handle the text
    WithSystemFallback,
    /// Only use explicitly configured primary fonts, no system fallback
    PrimaryFontsOnly,
}

use fontique::{self, Query, QueryFamily, QueryFont};
use swash::text::cluster::{CharCluster, CharInfo, Token};
use swash::text::{Language, Script};

/// Convert multiple character ranges to byte ranges efficiently in O(n) time
/// where n is the length of the text, regardless of the number of ranges.
pub(crate) fn char_ranges_to_byte_ranges(text: &str, char_ranges: &[core::ops::Range<usize>]) -> Vec<core::ops::Range<usize>> {
    if char_ranges.is_empty() {
        return Vec::new();
    }

    // Validate input ranges and create a sorted list of character positions we need to find
    let mut positions = Vec::new();
    let mut valid_ranges = Vec::new(); // Track which ranges are valid
    let mut valid_idx_counter = 0; // Separate counter for valid range indices

    for (original_idx, range) in char_ranges.iter().enumerate() {
        // Validate range: start should not be greater than end
        if range.start > range.end {
            // Consistent behavior: treat invalid ranges as empty (0..0) rather than panicking
            // This preserves the function contract while handling bad input gracefully
            valid_ranges.push(None);
            continue;
        }

        let valid_idx = valid_idx_counter;
        valid_idx_counter += 1;
        valid_ranges.push(Some(original_idx));

        positions.push((range.start, valid_idx, false)); // false = start position
        positions.push((range.end, valid_idx, true));    // true = end position
    }

    let num_valid_ranges = valid_idx_counter;

    // Use stable sort with secondary key to ensure deterministic behavior when positions are equal
    // Process start positions before end positions at the same character index
    positions.sort_by_key(|(pos, _range_idx, is_end)| (*pos, *is_end));

    // Results vector - track start and end positions separately to avoid invalid intermediate states
    let mut start_positions: Vec<Option<usize>> = vec![None; num_valid_ranges];
    let mut end_positions: Vec<Option<usize>> = vec![None; num_valid_ranges];
    let mut char_index = 0;
    let mut pos_iter = positions.iter().peekable();

    // Single pass through text to find all byte positions
    for (byte_index, _char) in text.char_indices() {
        // Record byte positions for all character positions that match current char_index
        while let Some((char_pos, range_idx, is_end)) = pos_iter.peek() {
            if *char_pos == char_index {
                if *is_end {
                    end_positions[*range_idx] = Some(byte_index);
                } else {
                    start_positions[*range_idx] = Some(byte_index);
                }
                pos_iter.next();
            } else {
                break;
            }
        }
        char_index += 1;
    }

    // Handle end-of-text positions (clamp to text boundaries)
    while let Some((char_pos, range_idx, is_end)) = pos_iter.next() {
        if *char_pos >= char_index {
            if *is_end {
                end_positions[*range_idx] = Some(text.len());
            } else {
                start_positions[*range_idx] = Some(text.len());
            }
        }
    }

    // Convert to concrete ranges, mapping back to original indices and handling invalid ranges
    let mut valid_idx = 0;
    valid_ranges.into_iter().map(|original_idx_opt| {
        match original_idx_opt {
            Some(_original_idx) => {
                let start = start_positions[valid_idx];
                let end = end_positions[valid_idx];
                valid_idx += 1;

                match (start, end) {
                    (Some(start_byte), Some(end_byte)) => start_byte..end_byte,
                    _ => {
                        // This shouldn't happen with valid input, but provides safe fallback
                        0..0 // Safe fallback for uninitialized range
                    }
                }
            }
            None => {
                // Invalid range that was skipped in release mode
                0..0 // Return empty range for invalid input
            }
        }
    }).collect()
}


/// Helper to advance parser and get next font selection result
fn advance_and_get_next_result<'a, 'b, I>(
    parser: &mut swash::text::cluster::Parser<I>,
    cluster: &mut CharCluster,
    strategy: &dyn crate::font_selection::FontSelectionStrategy,
    font_selector: &mut FontSelector<'a, 'b>,
    text: &str,
    text_range: &core::ops::Range<usize>,
    item_text: &str,
    current_char_index: &mut usize,
) -> Option<crate::font_selection::FontSelectionResult>
where
    I: Iterator<Item = Token> + Clone,
{
    if !parser.next(cluster) {
        return None; // End of text
    }

    // Calculate cluster character range without advancing current_char_index yet
    let cluster_byte_start = cluster.range().start as usize - text_range.start;
    let cluster_byte_end = cluster.range().end as usize - text_range.start;
    let cluster_char_count = item_text[cluster_byte_start..cluster_byte_end].chars().count();
    let cluster_char_range = *current_char_index..(*current_char_index + cluster_char_count);

    // IMPORTANT: Strategy receives the character range for THIS cluster BEFORE index advancement
    // This ensures consistent character positioning across strategy calls:
    // - Each strategy call gets the correct char_range for the cluster it's processing
    // - Index is advanced AFTER processing to point to the next cluster
    // - Next strategy call will receive the updated index as its starting position
    let result = strategy.select_font_for_cluster(
        cluster, font_selector, text, cluster_char_range
    );

    // Now advance the character index after strategy call
    // This maintains consistency: next cluster will start where this one ended
    *current_char_index += cluster_char_count;

    Some(result)
}

mod cache;

pub(crate) struct ShapeContext {
    shape_data_cache: LruCache<cache::ShapeDataKey, harfrust::ShaperData>,
    shape_instance_cache: LruCache<cache::ShapeInstanceId, harfrust::ShaperInstance>,
    shape_plan_cache: LruCache<cache::ShapePlanId, harfrust::ShapePlan>,
    unicode_buffer: Option<harfrust::UnicodeBuffer>,
    features: Vec<harfrust::Feature>,
}

impl Default for ShapeContext {
    fn default() -> Self {
        const MAX_ENTRIES: usize = 16;
        Self {
            shape_data_cache: LruCache::new(MAX_ENTRIES),
            shape_instance_cache: LruCache::new(MAX_ENTRIES),
            shape_plan_cache: LruCache::new(MAX_ENTRIES),
            unicode_buffer: Some(harfrust::UnicodeBuffer::new()),
            features: Vec::new(),
        }
    }
}

struct Item {
    style_index: u16,
    size: f32,
    script: Script,
    level: u8,
    locale: Option<Language>,
    variations: Resolved<FontVariation>,
    features: Resolved<FontFeature>,
    word_spacing: f32,
    letter_spacing: f32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn shape_text<'a, B: Brush>(
    rcx: &'a ResolveContext,
    mut fq: Query<'a>,
    styles: &'a [RangedStyle<B>],
    inline_boxes: &[InlineBox],
    infos: &[(CharInfo, u16)],
    levels: &[u8],
    scx: &mut ShapeContext,
    mut text: &str,
    layout: &mut Layout<B>,
    font_selection_strategy: &dyn crate::font_selection::FontSelectionStrategy,
) {
    // If we have both empty text and no inline boxes, shape with a fake space
    // to generate metrics that can be used to size a cursor.
    if text.is_empty() && inline_boxes.is_empty() {
        text = " ";
    }
    // Do nothing if there is no text or styles (there should always be a default style)
    if text.is_empty() || styles.is_empty() {
        // Process any remaining inline boxes whose index is greater than the length of the text
        for box_idx in 0..inline_boxes.len() {
            // Push the box to the list of items
            layout.data.push_inline_box(box_idx);
        }
        return;
    }

    // Setup mutable state for iteration
    let mut style = &styles[0];
    let mut item = Item {
        style_index: 0,
        size: style.font_style.font_size,
        level: levels.first().copied().unwrap_or(0),
        script: infos
            .iter()
            .map(|x| x.0.script())
            .find(|&script| real_script(script))
            .unwrap_or(Script::Latin),
        locale: style.font_style.locale,
        variations: style.font_style.font_variations,
        features: style.font_style.font_features,
        word_spacing: style.font_style.word_spacing,
        letter_spacing: style.font_style.letter_spacing,
    };
    let mut char_range = 0..0;
    let mut text_range = 0..0;

    let mut inline_box_iter = inline_boxes.iter().enumerate();
    let mut current_box = inline_box_iter.next();

    // Iterate over characters in the text
    for ((char_index, (byte_index, ch)), (info, style_index)) in
        text.char_indices().enumerate().zip(infos)
    {
        let mut break_run = false;
        let mut script = info.script();
        if !real_script(script) {
            script = item.script;
        }
        let level = levels.get(char_index).copied().unwrap_or(0);
        if item.style_index != *style_index {
            item.style_index = *style_index;
            style = &styles[*style_index as usize];
            if !nearly_eq(style.font_style.font_size, item.size)
                || style.font_style.locale != item.locale
                || style.font_style.font_variations != item.variations
                || style.font_style.font_features != item.features
                || !nearly_eq(style.font_style.letter_spacing, item.letter_spacing)
                || !nearly_eq(style.font_style.word_spacing, item.word_spacing)
            {
                break_run = true;
            }
        }

        if level != item.level || script != item.script {
            break_run = true;
        }

        // Check if there is an inline box at this index
        // Note:
        //   - We loop because there may be multiple boxes at this index
        //   - We do this *before* processing the text run because we need to know whether we should
        //     break the run due to the presence of an inline box.
        let mut deferred_boxes: Option<RangeInclusive<usize>> = None;
        while let Some((box_idx, inline_box)) = current_box {
            if inline_box.index == byte_index {
                break_run = true;
                if let Some(boxes) = &mut deferred_boxes {
                    deferred_boxes = Some((*boxes.start())..=box_idx);
                } else {
                    deferred_boxes = Some(box_idx..=box_idx);
                };
                // Update the current box to the next box
                current_box = inline_box_iter.next();
            } else {
                break;
            }
        }

        if break_run && !text_range.is_empty() {
            shape_item(
                &mut fq,
                rcx,
                styles,
                &item,
                scx,
                text,
                &text_range,
                &char_range,
                infos,
                layout,
                font_selection_strategy,
            );
            item.size = style.font_style.font_size;
            item.level = level;
            item.script = script;
            item.locale = style.font_style.locale;
            item.variations = style.font_style.font_variations;
            item.features = style.font_style.font_features;
            item.word_spacing = style.font_style.word_spacing;
            item.letter_spacing = style.font_style.letter_spacing;
            text_range.start = text_range.end;
            char_range.start = char_range.end;
        }

        if let Some(deferred_boxes) = deferred_boxes {
            for box_idx in deferred_boxes {
                layout.data.push_inline_box(box_idx);
            }
        }

        text_range.end += ch.len_utf8();
        char_range.end += 1;
    }

    if !text_range.is_empty() {
        shape_item(
            &mut fq,
            rcx,
            styles,
            &item,
            scx,
            text,
            &text_range,
            &char_range,
            infos,
            layout,
            font_selection_strategy,
        );
    }

    // Process any remaining inline boxes whose index is greater than the length of the text
    if let Some((box_idx, _inline_box)) = current_box {
        layout.data.push_inline_box(box_idx);
    }
    for (box_idx, _inline_box) in inline_box_iter {
        layout.data.push_inline_box(box_idx);
    }
}

fn shape_item<'a, B: Brush>(
    fq: &mut Query<'a>,
    rcx: &'a ResolveContext,
    styles: &'a [RangedStyle<B>],
    item: &Item,
    scx: &mut ShapeContext,
    text: &str,
    text_range: &core::ops::Range<usize>,
    char_range: &core::ops::Range<usize>,
    infos: &[(CharInfo, u16)],
    layout: &mut Layout<B>,
    strategy: &dyn crate::font_selection::FontSelectionStrategy,
) {
    let item_text = &text[text_range.clone()];
    let item_infos = &infos[char_range.start..char_range.end]; // Only process current item

    // Parse text into clusters of the current item
    let tokens =
        item_text
            .char_indices()
            .zip(item_infos)
            .map(|((offset, ch), (info, style_index))| Token {
                ch,
                offset: (text_range.start + offset) as u32,
                len: ch.len_utf8() as u8,
                info: *info,
                data: *style_index as u32,
            });

    let mut parser = swash::text::cluster::Parser::new(item.script, tokens);
    let mut cluster = CharCluster::new();

    // Get initial result for first cluster (preserving original algorithm structure)
    if !parser.next(&mut cluster) {
        return; // No clusters to process
    }

    // Track character position efficiently (fixes O(n²) regression)
    let mut current_char_index = char_range.start;

    // Calculate cluster character range without advancing current_char_index yet
    let cluster_byte_start = cluster.range().start as usize - text_range.start;
    let cluster_byte_end = cluster.range().end as usize - text_range.start;
    let cluster_char_count = item_text[cluster_byte_start..cluster_byte_end].chars().count();
    let cluster_char_range = current_char_index..(current_char_index + cluster_char_count);

    let style_index = cluster.user_data() as u16;

    // Create FontSelector ONCE per item (preserving ALL original performance optimizations)
    // Use correct item.script and item.locale (fixes functional regression)
    let fallback_mode = strategy.fallback_mode();
    let mut font_selector = FontSelector::new(
        fq, rcx, styles, style_index, item.script, item.locale, fallback_mode
    );

    let mut current_result = {
        // IMPORTANT: Strategy receives the character range for THIS cluster BEFORE index advancement
        // This ensures consistent character positioning - same contract as advance_and_get_next_result
        strategy.select_font_for_cluster(
            &mut cluster, &mut font_selector, text, cluster_char_range.clone()
        )
    };

    // Now advance the character index after strategy call (consistent with advance_and_get_next_result)
    current_char_index += cluster_char_count;

    // Main segmentation loop (preserves original performance characteristics)
    loop {
        match current_result {
            crate::font_selection::FontSelectionResult::UseFont(ref selected_font) => {
                // Clone the selected font to avoid ownership issues in the loop
                let current_selected_font = selected_font.clone();

                // Collect all consecutive clusters that use the same font (same as original)
                let segment_start_offset = cluster.range().start as usize - text_range.start;
                let mut segment_end_offset = cluster.range().end as usize - text_range.start;
                let mut end_of_text = false;

                // Inner loop: extend segment while clusters use same font
                loop {
                    let next_result = match advance_and_get_next_result(
                        &mut parser, &mut cluster, strategy, &mut font_selector,
                        text, text_range, item_text, &mut current_char_index
                    ) {
                        Some(result) => result,
                        None => {
                            // End of text - process final segment and then exit main loop
                            end_of_text = true;
                            break;
                        }
                    };

                    match next_result {
                        crate::font_selection::FontSelectionResult::UseFont(ref next_selected_font) if next_selected_font == &current_selected_font => {
                            // Same font - extend current segment
                            segment_end_offset = cluster.range().end as usize - text_range.start;
                        }
                        _ => {
                            // Different font or result type - end current segment
                            current_result = next_result;
                            break;
                        }
                    }
                }

                // Convert SelectedFont -> Font for shaping
                let current_font = FontData::new(
                    current_selected_font.font.blob.clone(),
                    current_selected_font.font.index,
                );

                // Shape the entire segment at once (preserves original performance)
                shape_segment_with_harfrust(
                    rcx, item, scx, text, text_range, char_range,
                    segment_start_offset..segment_end_offset, infos, &current_font, &current_selected_font.font.synthesis, layout
                );

                // If we reached end of text, exit main loop
                if end_of_text {
                    break;
                }
            }

            crate::font_selection::FontSelectionResult::UseFallbackSegments(segments) => {
                // Handle pre-segmented fallback results (new capability)

                // Validation: verify critical single-cluster contract
                if segments.is_empty() {
                    // Empty segments is a strategy error - treat as NoFont to preserve text positioning
                    // Convert to NoFont result instead of skipping text - preserves text positioning
                    current_result = crate::font_selection::FontSelectionResult::NoFont;
                    continue; // Re-process with NoFont result
                }

                // Note: Strategy contract assumes segments cover the current cluster exactly

                // Convert all character ranges to byte ranges efficiently in O(n) time
                let char_ranges: Vec<_> = segments.iter().map(|s| s.char_range.clone()).collect();
                let byte_ranges = char_ranges_to_byte_ranges(text, &char_ranges);

                for (segment, byte_range_in_text) in segments.iter().zip(byte_ranges.iter()) {
                    // Convert to offset within this text item
                    let segment_start_offset = byte_range_in_text.start - text_range.start;
                    let segment_end_offset = byte_range_in_text.end - text_range.start;

                    shape_segment_with_harfrust(
                        rcx, item, scx, text, text_range, char_range,
                        segment_start_offset..segment_end_offset, infos, &segment.font, &segment.synthesis, layout
                    );
                }

                // Move to next cluster after processing fallback segments
                // Note: The strategy is responsible for ensuring that the returned segments
                // cover the appropriate character ranges. We advance past the current cluster
                // since it has been processed via the fallback segments.
                if let Some(result) = advance_and_get_next_result(
                    &mut parser, &mut cluster, strategy, &mut font_selector,
                    text, text_range, item_text, &mut current_char_index
                ) {
                    current_result = result;
                } else {
                    break; // End of text
                }
            }

            crate::font_selection::FontSelectionResult::NoFont => {
                // Skip this cluster and move to next
                if let Some(result) = advance_and_get_next_result(
                    &mut parser, &mut cluster, strategy, &mut font_selector,
                    text, text_range, item_text, &mut current_char_index
                ) {
                    current_result = result;
                } else {
                    break; // End of text
                }
            }
        }

        // The main loop continues until we've processed all clusters.
        // For UseFont results, we continue with the current_result.
        // For UseFallbackSegments and NoFont, we already advanced to the next cluster above.
        //
        // The loop termination is handled by the "break" statements in each match branch
        // when parser.next() returns false (end of text).
    }

}

/// Shape a text segment with a specific font using harfrust.
/// This contains the core harfrust shaping logic, operating on segments rather than individual clusters
/// to preserve the performance characteristics of the original implementation.
fn shape_segment_with_harfrust<B: Brush>(
    rcx: &ResolveContext,
    item: &Item,
    scx: &mut ShapeContext,
    text: &str,
    text_range: &core::ops::Range<usize>,
    char_range: &core::ops::Range<usize>,
    segment_offset_range: core::ops::Range<usize>, // Byte offsets within the text item
    infos: &[(CharInfo, u16)],
    font: &FontData,
    synthesis: &fontique::Synthesis,
    layout: &mut Layout<B>,
) {
    let item_text = &text[text_range.clone()];
    let segment_text = &item_text[segment_offset_range.clone()];

    // Create harfrust font reference (using scoped borrow to avoid conflict)
    let font_index = font.index;
    let font_ref = match harfrust::FontRef::from_index(font.data.as_ref(), font_index) {
        Ok(font_ref) => font_ref,
        Err(_err) => {
            // Skip shaping if font is invalid - text segment will be missing
            // In debug builds, the original code had no logging here, keeping it clean
            return;
        }
    };

    // Create harfrust shaper data (cached)
    let font_data_id = font.data.id();
    let shaper_data = scx.shape_data_cache.entry(
        cache::ShapeDataKey::new(font_data_id, font_index),
        || harfrust::ShaperData::new(&font_ref),
    );

    // Create harfrust instance (cached)
    let instance = scx.shape_instance_cache.entry(
        cache::ShapeInstanceKey::new(
            font.data.id(),
            font.index,
            synthesis,
            rcx.variations(item.variations),
        ),
        || {
            harfrust::ShaperInstance::from_variations(
                &font_ref,
                variations_iter(synthesis, rcx.variations(item.variations)),
            )
        },
    );

    // Set up shaping parameters
    let direction = if item.level & 1 != 0 {
        harfrust::Direction::RightToLeft
    } else {
        harfrust::Direction::LeftToRight
    };
    let script = swash_convert::script_to_harfrust(item.script);
    let language = item
        .locale
        .and_then(|lang| lang.language().parse::<harfrust::Language>().ok());

    // Set up features
    scx.features.clear();
    for feature in rcx.features(item.features).unwrap_or(&[]) {
        scx.features.push(harfrust::Feature::new(
            harfrust::Tag::from_u32(feature.tag),
            feature.value as u32,
            ..,
        ));
    }

    // Create harfrust shaper
    let harf_shaper = shaper_data
        .shaper(&font_ref)
        .instance(Some(instance))
        .point_size(Some(item.size))
        .build();

    // Create shape plan (cached)
    let shaper_plan = scx.shape_plan_cache.entry(
        cache::ShapePlanKey::new(
            font.data.id(),
            font.index,
            synthesis,
            direction,
            script,
            language.clone(),
            &scx.features,
            rcx.variations(item.variations),
        ),
        || {
            harfrust::ShapePlan::new(
                &harf_shaper,
                direction,
                Some(script),
                language.as_ref(),
                &scx.features,
            )
        },
    );

    // Prepare harfrust buffer
    let mut buffer = core::mem::take(&mut scx.unicode_buffer).unwrap();
    buffer.clear();
    buffer.reserve(segment_text.len());

    // Add characters to buffer with proper cluster indices
    for (i, ch) in segment_text.chars().enumerate() {
        // Ensure that each cluster's index matches the index into `infos`. This is required
        // for efficient cluster lookup within `data.rs`.
        //
        // In other words, instead of using `buffer.push_str`, which iterates `segment_text`
        // with `char_indices`, push each char individually via `.chars` with a cluster index
        // that matches its `infos` counterpart. This allows us to lookup `infos` via cluster
        // index in `data.rs`.
        buffer.add(ch, i as u32);
    }

    buffer.set_direction(direction);
    buffer.set_script(script);
    if let Some(lang) = language {
        buffer.set_language(lang);
    }

    // Shape the text
    let glyph_buffer = harf_shaper.shape_with_plan(shaper_plan, buffer, &scx.features);

    // Calculate character indices for this segment within the item
    let segment_char_start_in_item = item_text[..segment_offset_range.start].chars().count();
    let segment_char_end_in_item = item_text[..segment_offset_range.end].chars().count();

    // Extract the proper CharInfo slice for this segment from the original infos
    let segment_char_start_absolute = char_range.start + segment_char_start_in_item;
    let segment_char_end_absolute = char_range.start + segment_char_end_in_item;

    // Get the CharInfo slice that corresponds to this text segment
    let segment_char_infos = &infos[segment_char_start_absolute..segment_char_end_absolute];

    // Push harfrust-shaped run for the entire segment with proper CharInfo data
    layout.data.push_run(
        font.clone(), // Clone for push_run
        item.size,
        synthesis.clone(),
        &glyph_buffer,
        item.level,
        item.style_index,
        item.word_spacing,
        item.letter_spacing,
        segment_text,
        segment_char_infos,
        (text_range.start + segment_offset_range.start)..(text_range.start + segment_offset_range.end),
        harf_shaper.coords(),
    );

    // Return buffer for reuse
    scx.unicode_buffer = Some(glyph_buffer.clear());
}

fn real_script(script: Script) -> bool {
    script != Script::Common && script != Script::Unknown && script != Script::Inherited
}

fn variations_iter<'a>(
    synthesis: &'a fontique::Synthesis,
    item: Option<&'a [FontVariation]>,
) -> impl Iterator<Item = harfrust::Variation> + 'a {
    synthesis
        .variation_settings()
        .iter()
        .map(|(tag, value)| harfrust::Variation {
            tag: *tag,
            value: *value,
        })
        .chain(
            item.unwrap_or(&[])
                .iter()
                .map(|variation| harfrust::Variation {
                    tag: harfrust::Tag::from_u32(variation.tag),
                    value: variation.value,
                }),
        )
}

pub(crate) struct FontSelector<'a, 'b> {
    query: &'b mut Query<'a>,
    fonts_id: Option<usize>,
    rcx: &'a ResolveContext,
    font_styles: Vec<Arc<crate::resolve::FontStyleData>>,
    style_index: u16,
    attrs: fontique::Attributes,
    variations: &'a [FontVariation],
    features: &'a [FontFeature],
    fallback_mode: FallbackMode,
}

impl<'a, 'b> FontSelector<'a, 'b> {
    pub(crate) fn new<B: Brush>(
        query: &'b mut Query<'a>,
        rcx: &'a ResolveContext,
        styles: &[RangedStyle<B>],
        style_index: u16,
        script: Script,
        locale: Option<Language>,
        fallback_mode: FallbackMode,
    ) -> Self {
        let style = &styles[style_index as usize].font_style;
        let fonts_id = style.font_stack.id();
        let fonts = rcx.stack(style.font_stack).unwrap_or(&[]);
        let attrs = fontique::Attributes {
            width: style.font_width,
            weight: style.font_weight,
            style: style.font_style,
        };
        let variations = rcx.variations(style.font_variations).unwrap_or(&[]);
        let features = rcx.features(style.font_features).unwrap_or(&[]);
        query.set_families(fonts.iter().copied());

        // Set up fallbacks based on mode (fixes performance regression)
        if let FallbackMode::WithSystemFallback = fallback_mode {
            let fb_script = swash_convert::script_to_fontique(script);
            let fb_language = locale.and_then(swash_convert::locale_to_fontique);
            query.set_fallbacks(fontique::FallbackKey::new(fb_script, fb_language.as_ref()));
        }
        // For PrimaryFontsOnly: don't set fallbacks at all - they're optional

        query.set_attributes(attrs);

        // Extract font styles from the RangedStyle array
        let font_styles: Vec<Arc<crate::resolve::FontStyleData>> = styles.iter()
            .map(|ranged_style| ranged_style.font_style.clone())
            .collect();

        Self {
            query,
            fonts_id: Some(fonts_id),
            rcx,
            font_styles,
            style_index,
            attrs,
            variations,
            features,
            fallback_mode,
        }
    }

    pub(crate) fn select_font(&mut self, cluster: &mut CharCluster) -> Option<SelectedFont> {
        let style_index = cluster.user_data() as u16;
        let is_emoji = cluster.info().is_emoji();
        if style_index != self.style_index || is_emoji || self.fonts_id.is_none() {
            self.style_index = style_index;
            let style = &self.font_styles[style_index as usize];

            let fonts_id = style.font_stack.id();
            let fonts = self.rcx.stack(style.font_stack).unwrap_or(&[]);
            let fonts = fonts.iter().copied().map(QueryFamily::Id);
            if is_emoji {
                use core::iter::once;
                let emoji_family = QueryFamily::Generic(fontique::GenericFamily::Emoji);
                self.query.set_families(fonts.chain(once(emoji_family)));
                self.fonts_id = None;
            } else if self.fonts_id != Some(fonts_id) {
                self.query.set_families(fonts);
                self.fonts_id = Some(fonts_id);
            }

            let attrs = fontique::Attributes {
                width: style.font_width,
                weight: style.font_weight,
                style: style.font_style,
            };
            if self.attrs != attrs {
                self.query.set_attributes(attrs);
                self.attrs = attrs;
            }
            self.variations = self.rcx.variations(style.font_variations).unwrap_or(&[]);
            self.features = self.rcx.features(style.font_features).unwrap_or(&[]);
        }

        // Fallbacks already configured in constructor (performance optimized)
        let mut selected_font = None;
        self.query.matches_with(|font| {
            use swash::text::cluster::Status as MapStatus;

            let Some(charmap) = font.charmap() else {
                return fontique::QueryStatus::Continue;
            };

            let map_status = cluster.map(|ch| {
                charmap
                    .map(ch)
                    .map(|g| {
                        // HACK: in reality, we're only using swash to compute
                        // coverage so we only care about whether the font
                        // has a mapping for a particular glyph. Any non-zero
                        // value indicates the existence of a glyph so we can
                        // simplify this without a fallible conversion from u32
                        // to u16.
                        (g != 0) as u16
                    })
                    .unwrap_or_default()
            });

            match map_status {
                MapStatus::Complete => {
                    selected_font = Some(font.into());
                    fontique::QueryStatus::Stop
                }
                MapStatus::Keep => {
                    selected_font = Some(font.into());
                    fontique::QueryStatus::Continue
                }
                MapStatus::Discard => {
                    // Behavior depends on fallback mode:
                    // - WithSystemFallback: select discarded fonts (original behavior)
                    // - PrimaryFontsOnly: don't select discarded fonts (let strategy handle)
                    if self.fallback_mode == FallbackMode::WithSystemFallback && selected_font.is_none() {
                        selected_font = Some(font.into());
                    }
                    fontique::QueryStatus::Continue
                }
            }
        });
        selected_font
    }
}

#[derive(Debug, Clone)]
pub struct SelectedFont {
    pub font: QueryFont,
}

impl From<&QueryFont> for SelectedFont {
    fn from(font: &QueryFont) -> Self {
        Self { font: font.clone() }
    }
}

impl PartialEq for SelectedFont {
    fn eq(&self, other: &Self) -> bool {
        self.font.family == other.font.family && self.font.synthesis == other.font.synthesis
    }
}
