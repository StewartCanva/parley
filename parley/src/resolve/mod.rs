// Copyright 2021 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Resolution of dynamic properties within a context.

pub(crate) mod range;
pub(crate) mod tree;

pub(crate) use range::RangedStyleBuilder;

use alloc::{vec, vec::Vec};

use super::style::{
    Brush, FontFamily, FontFeature, FontSettings, FontStack, FontStyle, FontVariation, FontWeight,
    FontWidth, StyleProperty,
};
use crate::font::FontContext;
use crate::style::TextStyle;
use crate::util::nearly_eq;
use crate::{LineHeight, OverflowWrap, WordBreakStrength, layout};
use core::borrow::Borrow;
use core::ops::Range;
use fontique::FamilyId;
use swash::text::Language;
use std::sync::Arc;

/// Style with an associated range.
#[derive(Debug, Clone)]
pub(crate) struct RangedStyle<B: Brush> {
    pub(crate) font_style: Arc<FontStyleData>,
    pub(crate) render_style: RenderStyleData<B>,
    pub(crate) range: Range<usize>,
}

impl<B: Brush> RangedStyle<B> {
    /// Check if a property would change this style.
    pub(crate) fn check(&self, property: &ResolvedProperty<B>) -> bool {
        // Create a temporary ResolvedStyle for checking
        let temp_style = self.as_resolved_style();
        temp_style.check(property)
    }

    /// Apply a property to this style, creating new Arc if font properties change.
    pub(crate) fn apply(&mut self, property: ResolvedProperty<B>) {
        // Check if this affects font properties
        if self.affects_font_style(&property) {
            // Create a new FontStyleData with the property applied
            let mut temp_style = self.as_resolved_style();
            temp_style.apply(property);
            self.font_style = Arc::new(FontStyleData::from_resolved_style(&temp_style));
            self.render_style = RenderStyleData::from_resolved_style(&temp_style);
        } else {
            // Only affects render properties - apply directly
            let mut temp_style = self.as_resolved_style();
            temp_style.apply(property);
            self.render_style = RenderStyleData::from_resolved_style(&temp_style);
        }
    }

    /// Create a temporary ResolvedStyle for operations that need the full style.
    fn as_resolved_style(&self) -> ResolvedStyle<B> {
        ResolvedStyle {
            font_stack: self.font_style.font_stack,
            font_size: self.font_style.font_size,
            font_width: self.font_style.font_width,
            font_style: self.font_style.font_style,
            font_weight: self.font_style.font_weight,
            font_variations: self.font_style.font_variations,
            font_features: self.font_style.font_features,
            locale: self.font_style.locale,
            word_spacing: self.font_style.word_spacing,
            letter_spacing: self.font_style.letter_spacing,
            brush: self.render_style.brush.clone(),
            underline: self.render_style.underline.clone(),
            strikethrough: self.render_style.strikethrough.clone(),
            line_height: self.render_style.line_height,
            word_break: self.render_style.word_break,
            overflow_wrap: self.render_style.overflow_wrap,
        }
    }

    /// Check if a property affects font style (vs only render style).
    fn affects_font_style(&self, property: &ResolvedProperty<B>) -> bool {
        matches!(property,
            ResolvedProperty::FontStack(_) |
            ResolvedProperty::FontSize(_) |
            ResolvedProperty::FontWidth(_) |
            ResolvedProperty::FontStyle(_) |
            ResolvedProperty::FontWeight(_) |
            ResolvedProperty::FontVariations(_) |
            ResolvedProperty::FontFeatures(_) |
            ResolvedProperty::Locale(_) |
            ResolvedProperty::WordSpacing(_) |
            ResolvedProperty::LetterSpacing(_)
        )
    }

    /// Convert to a layout::Style for rendering.
    pub(crate) fn as_layout_style(&self) -> layout::Style<B> {
        layout::Style {
            brush: self.render_style.brush.clone(),
            underline: self.render_style.underline.as_layout_decoration(&self.render_style.brush),
            strikethrough: self.render_style.strikethrough.as_layout_decoration(&self.render_style.brush),
            line_height: self.render_style.line_height.resolve(self.font_style.font_size),
            overflow_wrap: self.render_style.overflow_wrap,
        }
    }
}

/// Font-only style data (no brush/rendering properties).
/// Used for font selection which doesn't need rendering information.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FontStyleData {
    /// Font stack.
    pub(crate) font_stack: Resolved<FamilyId>,
    /// Font size.
    pub(crate) font_size: f32,
    /// Font width.
    pub(crate) font_width: FontWidth,
    /// Font style.
    pub(crate) font_style: FontStyle,
    /// Font weight.
    pub(crate) font_weight: FontWeight,
    /// Font variation settings.
    pub(crate) font_variations: Resolved<FontVariation>,
    /// Font feature settings.
    pub(crate) font_features: Resolved<FontFeature>,
    /// Locale.
    pub(crate) locale: Option<Language>,
    /// Extra spacing between words.
    pub(crate) word_spacing: f32,
    /// Extra spacing between letters.
    pub(crate) letter_spacing: f32,
}

impl FontStyleData {
    /// Extract font-only data from a ResolvedStyle (dropping brush/rendering properties).
    pub(crate) fn from_resolved_style<B: Brush>(style: &ResolvedStyle<B>) -> Self {
        Self {
            font_stack: style.font_stack,
            font_size: style.font_size,
            font_width: style.font_width,
            font_style: style.font_style,
            font_weight: style.font_weight,
            font_variations: style.font_variations,
            font_features: style.font_features,
            locale: style.locale,
            word_spacing: style.word_spacing,
            letter_spacing: style.letter_spacing,
        }
    }
}

/// Rendering-only style data (no font properties).
/// Used for layout and painting which don't need font selection information.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RenderStyleData<B: Brush> {
    /// Brush for rendering text.
    pub(crate) brush: B,
    /// Underline decoration.
    pub(crate) underline: ResolvedDecoration<B>,
    /// Strikethrough decoration.
    pub(crate) strikethrough: ResolvedDecoration<B>,
    /// Line height.
    pub(crate) line_height: LineHeight,
    /// Control over where words can wrap.
    pub(crate) word_break: WordBreakStrength,
    /// Control over "emergency" line-breaking.
    pub(crate) overflow_wrap: OverflowWrap,
}

impl<B: Brush> RenderStyleData<B> {
    /// Extract rendering-only data from a ResolvedStyle (dropping font properties).
    pub(crate) fn from_resolved_style(style: &ResolvedStyle<B>) -> Self {
        Self {
            brush: style.brush.clone(),
            underline: style.underline.clone(),
            strikethrough: style.strikethrough.clone(),
            line_height: style.line_height,
            word_break: style.word_break,
            overflow_wrap: style.overflow_wrap,
        }
    }
}


#[derive(Clone)]
struct RangedProperty<B: Brush> {
    property: ResolvedProperty<B>,
    range: Range<usize>,
}

/// Handle for a managed property.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) struct Resolved<T> {
    index: usize,
    _phantom: core::marker::PhantomData<T>,
}

impl<T> Default for Resolved<T> {
    fn default() -> Self {
        Self {
            index: !0,
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<T> Resolved<T> {
    pub(crate) fn id(&self) -> usize {
        self.index
    }
}

#[derive(Clone)]
struct Cache<T> {
    /// Items in the cache. May contain sequences.
    items: Vec<T>,
    /// Each entry represents a range of items in `data`.
    entries: Vec<(usize, usize)>,
}

impl<T> Default for Cache<T> {
    fn default() -> Self {
        Self {
            items: vec![],
            entries: vec![],
        }
    }
}

impl<T: Clone + PartialEq> Cache<T> {
    pub(crate) fn clear(&mut self) {
        self.items.clear();
        self.entries.clear();
    }

    pub(crate) fn insert(&mut self, items: &[T]) -> Resolved<T> {
        for (i, entry) in self.entries.iter().enumerate() {
            let range = entry.0..entry.1;
            if range.len() != items.len() {
                continue;
            }
            if let Some(existing) = self.items.get(range) {
                if existing == items {
                    return Resolved {
                        index: i,
                        _phantom: core::marker::PhantomData,
                    };
                }
            }
        }
        let index = self.entries.len();
        let start = self.items.len();
        self.items.extend(items.iter().cloned());
        let end = self.items.len();
        self.entries.push((start, end));
        Resolved {
            index,
            _phantom: core::marker::PhantomData,
        }
    }

    pub(crate) fn get(&self, handle: Resolved<T>) -> Option<&[T]> {
        let (start, end) = *self.entries.get(handle.index)?;
        self.items.get(start..end)
    }
}

/// Context for managing dynamic properties during layout.
#[derive(Clone, Default)]
pub(crate) struct ResolveContext {
    families: Cache<FamilyId>,
    variations: Cache<FontVariation>,
    features: Cache<FontFeature>,
    tmp_families: Vec<FamilyId>,
    tmp_variations: Vec<FontVariation>,
    tmp_features: Vec<FontFeature>,
}

impl ResolveContext {
    pub(crate) fn resolve_property<B: Brush>(
        &mut self,
        fcx: &mut FontContext,
        property: &StyleProperty<'_, B>,
        scale: f32,
    ) -> ResolvedProperty<B> {
        use ResolvedProperty::*;
        match property {
            StyleProperty::FontStack(value) => FontStack(self.resolve_stack::<B>(fcx, value)),
            StyleProperty::FontSize(value) => FontSize(*value * scale),
            StyleProperty::FontWidth(value) => FontWidth(*value),
            StyleProperty::FontStyle(value) => FontStyle(*value),
            StyleProperty::FontWeight(value) => FontWeight(*value),
            StyleProperty::FontVariations(value) => FontVariations(self.resolve_variations(value)),
            StyleProperty::FontFeatures(value) => FontFeatures(self.resolve_features(value)),
            StyleProperty::Locale(value) => Locale(value.map(Language::parse).flatten()),
            StyleProperty::Brush(value) => Brush(value.clone()),
            StyleProperty::Underline(value) => Underline(*value),
            StyleProperty::UnderlineOffset(value) => UnderlineOffset(value.map(|x| x * scale)),
            StyleProperty::UnderlineSize(value) => UnderlineSize(value.map(|x| x * scale)),
            StyleProperty::UnderlineBrush(value) => UnderlineBrush(value.clone()),
            StyleProperty::Strikethrough(value) => Strikethrough(*value),
            StyleProperty::StrikethroughOffset(value) => {
                StrikethroughOffset(value.map(|x| x * scale))
            }
            StyleProperty::StrikethroughSize(value) => StrikethroughSize(value.map(|x| x * scale)),
            StyleProperty::StrikethroughBrush(value) => StrikethroughBrush(value.clone()),
            StyleProperty::LineHeight(value) => LineHeight(value.scale(scale)),
            StyleProperty::WordSpacing(value) => WordSpacing(*value * scale),
            StyleProperty::LetterSpacing(value) => LetterSpacing(*value * scale),
            StyleProperty::WordBreak(value) => WordBreak(*value),
            StyleProperty::OverflowWrap(value) => OverflowWrap(*value),
        }
    }

    pub(crate) fn resolve_entire_style_set<B: Brush>(
        &mut self,
        fcx: &mut FontContext,
        raw_style: &TextStyle<'_, B>,
        scale: f32,
    ) -> ResolvedStyle<B> {
        ResolvedStyle {
            font_stack: self.resolve_stack::<B>(fcx, &raw_style.font_stack),
            font_size: raw_style.font_size * scale,
            font_width: raw_style.font_width,
            font_style: raw_style.font_style,
            font_weight: raw_style.font_weight,
            font_variations: self.resolve_variations(&raw_style.font_variations),
            font_features: self.resolve_features(&raw_style.font_features),
            locale: raw_style.locale.and_then(Language::parse),
            brush: raw_style.brush.clone(),
            underline: ResolvedDecoration {
                enabled: raw_style.has_underline,
                offset: raw_style.underline_offset.map(|x| x * scale),
                size: raw_style.underline_size.map(|x| x * scale),
                brush: raw_style.underline_brush.clone(),
            },
            strikethrough: ResolvedDecoration {
                enabled: raw_style.has_strikethrough,
                offset: raw_style.strikethrough_offset.map(|x| x * scale),
                size: raw_style.strikethrough_size.map(|x| x * scale),
                brush: raw_style.strikethrough_brush.clone(),
            },
            line_height: raw_style.line_height.scale(scale),
            word_spacing: raw_style.word_spacing * scale,
            letter_spacing: raw_style.letter_spacing * scale,
            word_break: raw_style.word_break,
            overflow_wrap: raw_style.overflow_wrap,
        }
    }

    /// Resolves a font stack.
    pub(crate) fn resolve_stack<B: crate::style::Brush>(
        &mut self,
        fcx: &mut FontContext,
        stack: &FontStack<'_>,
    ) -> Resolved<FamilyId> {
        self.tmp_families.clear();
        match stack {
            FontStack::Source(source) => {
                for family in FontFamily::parse_list(source) {
                    match family {
                        FontFamily::Named(name) => {
                            if let Some(family) = fcx.collection.family_by_name(&name) {
                                self.tmp_families.push(family.id());
                            }
                        }
                        FontFamily::Generic(family) => {
                            self.tmp_families
                                .extend(fcx.collection.generic_families(family));
                        }
                    }
                }
            }
            FontStack::Single(family) => match family {
                FontFamily::Named(name) => {
                    if let Some(family) = fcx.collection.family_by_name(name) {
                        self.tmp_families.push(family.id());
                    }
                }
                FontFamily::Generic(family) => {
                    self.tmp_families
                        .extend(fcx.collection.generic_families(*family));
                }
            },
            FontStack::List(families) => {
                let families: &[FontFamily<'_>] = families.borrow();
                for family in families {
                    match family {
                        FontFamily::Named(name) => {
                            if let Some(family) = fcx.collection.family_by_name(name) {
                                self.tmp_families.push(family.id());
                            }
                        }
                        FontFamily::Generic(family) => {
                            self.tmp_families
                                .extend(fcx.collection.generic_families(*family));
                        }
                    }
                }
            }
        }
        let resolved = self.families.insert(&self.tmp_families);
        self.tmp_families.clear();
        resolved
    }

    /// Resolves font variation settings.
    pub(crate) fn resolve_variations(
        &mut self,
        variations: &FontSettings<'_, FontVariation>,
    ) -> Resolved<FontVariation> {
        match variations {
            FontSettings::Source(source) => {
                self.tmp_variations.clear();
                self.tmp_variations
                    .extend(FontVariation::parse_list(source));
            }
            FontSettings::List(settings) => {
                self.tmp_variations.clear();
                self.tmp_variations.extend_from_slice(settings);
            }
        }
        if self.tmp_variations.is_empty() {
            return Resolved::default();
        }
        self.tmp_variations.sort_by(|a, b| a.tag.cmp(&b.tag));
        let resolved = self.variations.insert(&self.tmp_variations);
        self.tmp_variations.clear();
        resolved
    }

    /// Resolves font feature settings.
    pub(crate) fn resolve_features(
        &mut self,
        features: &FontSettings<'_, FontFeature>,
    ) -> Resolved<FontFeature> {
        match features {
            FontSettings::Source(source) => {
                self.tmp_features.clear();
                self.tmp_features.extend(FontFeature::parse_list(source));
            }
            FontSettings::List(settings) => {
                self.tmp_features.clear();
                self.tmp_features.extend_from_slice(settings);
            }
        }
        if self.tmp_features.is_empty() {
            return Resolved::default();
        }
        self.tmp_features.sort_by(|a, b| a.tag.cmp(&b.tag));
        let resolved = self.features.insert(&self.tmp_features);
        self.tmp_features.clear();
        resolved
    }

    /// Returns the list of font families for the specified handle.
    pub(crate) fn stack(&self, stack: Resolved<FamilyId>) -> Option<&[FamilyId]> {
        self.families.get(stack)
    }

    /// Returns the list of font variations for the specified handle.
    pub(crate) fn variations(
        &self,
        variations: Resolved<FontVariation>,
    ) -> Option<&[FontVariation]> {
        self.variations.get(variations)
    }

    /// Returns the list of font features for the specified handle.
    pub(crate) fn features(&self, features: Resolved<FontFeature>) -> Option<&[FontFeature]> {
        self.features.get(features)
    }

    /// Clears the resources in the context.
    pub(crate) fn clear(&mut self) {
        self.families.clear();
        self.variations.clear();
        self.features.clear();
    }
}

/// Style property with resolved resources.
#[derive(Clone, PartialEq)]
pub(crate) enum ResolvedProperty<B: Brush> {
    /// Font stack.
    FontStack(Resolved<FamilyId>),
    /// Font size.
    FontSize(f32),
    /// Font width.
    FontWidth(FontWidth),
    /// Font style.
    FontStyle(FontStyle),
    /// Font weight.
    FontWeight(FontWeight),
    /// Font variation settings.
    FontVariations(Resolved<FontVariation>),
    /// Font feature settings.
    FontFeatures(Resolved<FontFeature>),
    /// Locale.
    Locale(Option<Language>),
    /// Brush for rendering text.
    Brush(B),
    /// Underline decoration.
    Underline(bool),
    /// Offset of the underline decoration.
    UnderlineOffset(Option<f32>),
    /// Size of the underline decoration.
    UnderlineSize(Option<f32>),
    /// Brush for rendering the underline decoration.
    UnderlineBrush(Option<B>),
    /// Strikethrough decoration.
    Strikethrough(bool),
    /// Offset of the strikethrough decoration.
    StrikethroughOffset(Option<f32>),
    /// Size of the strikethrough decoration.
    StrikethroughSize(Option<f32>),
    /// Brush for rendering the strikethrough decoration.
    StrikethroughBrush(Option<B>),
    /// Line height.
    LineHeight(LineHeight),
    /// Extra spacing between words.
    WordSpacing(f32),
    /// Extra spacing between letters.
    LetterSpacing(f32),
    /// Control over where words can wrap.
    WordBreak(WordBreakStrength),
    /// Control over "emergency" line-breaking.
    OverflowWrap(OverflowWrap),
}

/// Flattened group of style properties.
#[derive(Clone, PartialEq, Debug, Default)]
pub(crate) struct ResolvedStyle<B: Brush> {
    /// Font stack.
    pub(crate) font_stack: Resolved<FamilyId>,
    /// Font size.
    pub(crate) font_size: f32,
    /// Font width.
    pub(crate) font_width: FontWidth,
    /// Font style.
    pub(crate) font_style: FontStyle,
    /// Font weight.
    pub(crate) font_weight: FontWeight,
    /// Font variation settings.
    pub(crate) font_variations: Resolved<FontVariation>,
    /// Font feature settings.
    pub(crate) font_features: Resolved<FontFeature>,
    /// Locale.
    pub(crate) locale: Option<Language>,
    /// Brush for rendering text.
    pub(crate) brush: B,
    /// Underline decoration.
    pub(crate) underline: ResolvedDecoration<B>,
    /// Strikethrough decoration.
    pub(crate) strikethrough: ResolvedDecoration<B>,
    /// Line height.
    pub(crate) line_height: LineHeight,
    /// Extra spacing between words.
    pub(crate) word_spacing: f32,
    /// Extra spacing between letters.
    pub(crate) letter_spacing: f32,
    /// Control over where words can wrap.
    pub(crate) word_break: WordBreakStrength,
    /// Control over "emergency" line-breaking.
    pub(crate) overflow_wrap: OverflowWrap,
}

impl<B: Brush> ResolvedStyle<B> {
    /// Applies the specified property to this style.
    pub(crate) fn apply(&mut self, property: ResolvedProperty<B>) {
        use ResolvedProperty::*;
        match property {
            FontStack(value) => self.font_stack = value,
            FontSize(value) => self.font_size = value,
            FontWidth(value) => self.font_width = value,
            FontStyle(value) => self.font_style = value,
            FontWeight(value) => self.font_weight = value,
            FontVariations(value) => self.font_variations = value,
            FontFeatures(value) => self.font_features = value,
            Locale(value) => self.locale = value,
            Brush(value) => self.brush = value,
            Underline(value) => self.underline.enabled = value,
            UnderlineOffset(value) => self.underline.offset = value,
            UnderlineSize(value) => self.underline.size = value,
            UnderlineBrush(value) => self.underline.brush = value,
            Strikethrough(value) => self.strikethrough.enabled = value,
            StrikethroughOffset(value) => self.strikethrough.offset = value,
            StrikethroughSize(value) => self.strikethrough.size = value,
            StrikethroughBrush(value) => self.strikethrough.brush = value,
            LineHeight(value) => self.line_height = value,
            WordSpacing(value) => self.word_spacing = value,
            LetterSpacing(value) => self.letter_spacing = value,
            WordBreak(value) => self.word_break = value,
            OverflowWrap(value) => self.overflow_wrap = value,
        }
    }

    pub(crate) fn check(&self, property: &ResolvedProperty<B>) -> bool {
        use ResolvedProperty::*;
        match property {
            FontStack(value) => self.font_stack == *value,
            FontSize(value) => nearly_eq(self.font_size, *value),
            FontWidth(value) => self.font_width == *value,
            FontStyle(value) => self.font_style == *value,
            FontWeight(value) => self.font_weight == *value,
            FontVariations(value) => self.font_variations == *value,
            FontFeatures(value) => self.font_features == *value,
            Locale(value) => self.locale == *value,
            Brush(value) => self.brush == *value,
            Underline(value) => self.underline.enabled == *value,
            UnderlineOffset(value) => self.underline.offset == *value,
            UnderlineSize(value) => self.underline.size == *value,
            UnderlineBrush(value) => self.underline.brush == *value,
            Strikethrough(value) => self.strikethrough.enabled == *value,
            StrikethroughOffset(value) => self.strikethrough.offset == *value,
            StrikethroughSize(value) => self.strikethrough.size == *value,
            StrikethroughBrush(value) => self.strikethrough.brush == *value,
            LineHeight(value) => self.line_height.nearly_eq(value),
            WordSpacing(value) => nearly_eq(self.word_spacing, *value),
            LetterSpacing(value) => nearly_eq(self.letter_spacing, *value),
            WordBreak(value) => self.word_break == *value,
            OverflowWrap(value) => self.overflow_wrap == *value,
        }
    }

}

/// Underline or strikethrough decoration.
#[derive(Clone, PartialEq, Default, Debug)]
pub(crate) struct ResolvedDecoration<B: Brush> {
    /// True if the decoration is enabled.
    pub(crate) enabled: bool,
    /// Offset of the decoration from the baseline.
    pub(crate) offset: Option<f32>,
    /// Thickness of the decoration stroke.
    pub(crate) size: Option<f32>,
    /// Brush for the decoration.
    pub(crate) brush: Option<B>,
}

impl<B: Brush> ResolvedDecoration<B> {
    /// Convert into a layout Decoration (filtering out disabled decorations)
    pub(crate) fn as_layout_decoration(&self, default_brush: &B) -> Option<layout::Decoration<B>> {
        if self.enabled {
            Some(layout::Decoration {
                brush: self.brush.clone().unwrap_or_else(|| default_brush.clone()),
                offset: self.offset,
                size: self.size,
            })
        } else {
            None
        }
    }
}
