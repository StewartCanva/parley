// Copyright 2021 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fontique::Collection;
use fontique::SourceCache;
use std::sync::Arc;

/// A font database/cache (wrapper around a Fontique [`Collection`] and [`SourceCache`]).
///
/// This type is designed to be a global resource with only one per-application (or per-thread).
/// A font database/cache and font selection configuration.
pub struct FontContext {
    pub collection: Collection,
    pub source_cache: SourceCache,
    font_selection_strategy: Option<Arc<dyn crate::font_selection::FontSelectionStrategy + Send + Sync>>,
}

impl Default for FontContext {
    fn default() -> Self {
        Self {
            collection: Collection::default(),
            source_cache: SourceCache::default(),
            font_selection_strategy: None,
        }
    }
}

impl Clone for FontContext {
    /// Clone the FontContext, including any custom font selection strategy.
    ///
    /// Custom strategies are properly cloned, so the cloned FontContext will
    /// maintain the same font selection behavior as the original.
    fn clone(&self) -> Self {
        Self {
            collection: self.collection.clone(),
            source_cache: self.source_cache.clone(),
            font_selection_strategy: self.font_selection_strategy.clone(),
        }
    }
}

impl FontContext {
    /// Create a new `FontContext`, discovering system fonts if available.
    pub fn new() -> Self {
        Default::default()
    }

    /// Set a custom font selection strategy.
    ///
    /// The strategy will be properly cloned if you clone the FontContext, so both
    /// instances will maintain the same font selection behavior.
    pub fn set_font_selection_strategy(
        &mut self,
        strategy: impl crate::font_selection::FontSelectionStrategy + Send + Sync + 'static,
    ) {
        self.font_selection_strategy = Some(Arc::new(strategy));
    }


    /// Execute a function with both a fontique query and the font selection strategy.
    pub(crate) fn with_query_and_strategy<R>(&mut self, f: impl FnOnce(fontique::Query<'_>, &dyn crate::font_selection::FontSelectionStrategy) -> R) -> R {
        if self.font_selection_strategy.is_none() {
            self.font_selection_strategy = Some(Arc::new(crate::font_selection::DefaultFontSelectionStrategy::new()));
        }
        let query = self.collection.query(&mut self.source_cache);
        let strategy = self.font_selection_strategy.as_ref().unwrap();
        f(query, strategy.as_ref())
    }
}
