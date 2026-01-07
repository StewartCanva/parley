// Copyright 2024 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A simple example that lays out some text using Parley, rasterises the glyph using Swash
//! and and then renders it into a PNG using the `image` crate.

#![expect(clippy::cast_possible_truncation, reason = "Deferred")]

use image::codecs::png::PngEncoder;
use image::{self, Pixel, Rgba, RgbaImage};
use parley::layout::{Alignment, Glyph, GlyphRun, Layout, PositionedLayoutItem};
use parley::style::{FontStack, FontStyle, FontWeight, StyleProperty, TextStyle};
use parley::{fontique, AlignmentOptions, CanvaFontSelectionStrategy, FontContext, FontData, FontFamily, InlineBox, LayoutContext, LineHeight};
use std::fs::File;
use std::sync::Arc;
use swash::FontRef;
use swash::scale::image::Content;
use swash::scale::{Render, ScaleContext, Scaler, Source, StrikeWith};
use swash::zeno;
use zeno::{Format, Vector};
use parley::fontique::Blob;

#[derive(Clone, Copy, Debug, PartialEq)]
struct ColorBrush {
    color: Rgba<u8>,
}

impl Default for ColorBrush {
    fn default() -> Self {
        Self {
            color: Rgba([0, 0, 0, 255]),
        }
    }
}

fn main() {
    // The display scale for HiDPI rendering
    let display_scale = 1.0;
    // Whether to automatically align the output to pixel boundaries, to avoid blurry text.
    let quantize = true;
    // The width for line wrapping
    let max_advance = Some(200.0 * display_scale);
    // Colours for rendering
    let text_color = Rgba([0, 0, 0, 255]);
    let bg_color = Rgba([255, 255, 255, 255]);
    // Padding around the output image
    let padding = 20;

    // !!!!

    // Create font context with custom fonts
    let mut font_ctx = FontContext::new();

    // Load the three test fonts
    let font_ag_data = std::fs::read("/Users/conor/work/font_splitting/FontAG.ttf")
        .expect("Failed to load FontAG");
    let font_hv_data = std::fs::read("/Users/conor/work/font_splitting/FontHV.ttf")
        .expect("Failed to load FontHV");
    let font_wz_data = std::fs::read("/Users/conor/work/font_splitting/FontWZ.ttf")
        .expect("Failed to load FontWZ");

    // Create blob objects that will be registered
    let blob_ag = Blob::new(Arc::new(font_ag_data));
    let blob_hv = Blob::new(Arc::new(font_hv_data));
    let blob_wz = Blob::new(Arc::new(font_wz_data));

    // Register fonts with collection
    font_ctx.collection.register_fonts(blob_ag.clone(), None);
    font_ctx.collection.register_fonts(blob_hv.clone(), None);
    font_ctx.collection.register_fonts(blob_wz.clone(), None);

    // Verify fonts are registered
    assert!(
        font_ctx.collection.family_id("FontAG").is_some(),
        "FontAG not registered"
    );
    assert!(
        font_ctx.collection.family_id("FontHV").is_some(),
        "FontHV not registered"
    );
    assert!(
        font_ctx.collection.family_id("FontWZ").is_some(),
        "FontWZ not registered"
    );

    // Create Font objects for the strategy using the same blobs
    let font_ag = FontData::new(blob_ag, 0);
    let font_hv = FontData::new(blob_hv, 0);
    //let font_wz = FontData::new(blob_wz, 0);

    // Set up Canva font selection strategy
    let mut canva_strategy = CanvaFontSelectionStrategy::new();

    // Configure unicode ranges with different synthesis to test the functionality:
    // FontHV for K-V with regular synthesis
    // TODO(conor) - When this range is extended to ..0x59 (reaching X), the test doesn't fail.
    //  Why? FontHV doesn't support the W or X glyphs. Maybe its just producing NoFont for each impacted cluster, which is fine?
    canva_strategy.add_unicode_range_with_synthesis(
        0x4B..0x55,
        font_hv.clone(),
        FontWeight::NORMAL,
        FontStyle::Normal,
        fontique::Synthesis::default(),
    ); // J-S uppercase, regular
    //canva_strategy.add_unicode_range_with_synthesis(0x6A..0x74, font_hv.clone(), fontique::Synthesis::default()); // j-s lowercase, regular

    // FontWZ for Y-Z with regular synthesis
    /*canva_strategy.add_unicode_range_with_synthesis(
        0x59..0x5B,
        font_wz.clone(),
        fontique::Synthesis::default(),
    );*/ // W-Z uppercase, regular
    //canva_strategy.add_unicode_range_with_synthesis(0x77..0x7B, font_wz.clone(), fontique::Synthesis::default()); // w-z lowercase, regular

    assert_eq!(
        canva_strategy.len(),
        1,
        "Should have 2 unicode ranges configured"
    );

    // Note: In the future, you could add synthesis variants like:
    // let bold_synthesis = create_bold_synthesis();
    // canva_strategy.add_unicode_range_with_synthesis(0x4A..0x54, font_gs_bold, bold_synthesis);
    // This would allow different fonts/synthesis for the same unicode ranges based on style context

    // Set the strategy on font context (where it belongs!)
    font_ctx.set_font_selection_strategy(canva_strategy);

    // Create layout context
    let mut layout_cx: LayoutContext<ColorBrush> = LayoutContext::new();

    // Test with full alphabet - FontAG should only handle A-G
    //let text = "éABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    // Create layout with FontAG as primary font
    let mut builder = layout_cx.ranged_builder(&mut font_ctx, text, 1.0, true);
    builder.push_default(StyleProperty::FontStack(FontStack::Single(
        FontFamily::Named(std::borrow::Cow::Borrowed("FontAG")),
    )));
    builder.push_default(StyleProperty::FontSize(16.0));

    let mut layout = builder.build(text);
    layout.break_all_lines(None);

    // !!!!

    // Setup some Parley text styles
    let text_brush = ColorBrush { color: text_color };
    let brush_style = StyleProperty::Brush(text_brush);
    let font_stack = FontStack::from("system-ui");

    // Creates a RangedBuilder
    let mut builder = layout_cx.ranged_builder(&mut font_ctx, &text, display_scale, quantize);

    // Set default text colour styles (set foreground text color)
    builder.push_default(brush_style);

    // Set default font family
    builder.push_default(font_stack);
    builder.push_default(LineHeight::FontSizeRelative(1.3));
    builder.push_default(StyleProperty::FontSize(16.0));

    // Build the builder into a Layout
    // let mut layout: Layout<ColorBrush> = builder.build(&text);
    let mut layout: Layout<ColorBrush> = builder.build(&text);

    // Perform layout (including bidi resolution and shaping) with start alignment
    layout.break_all_lines(max_advance);
    layout.align(max_advance, Alignment::Start, AlignmentOptions::default());

    // Create image to render into
    let width = layout.width().ceil() as u32 + (padding * 2);
    let height = layout.height().ceil() as u32 + (padding * 2);
    let mut img = RgbaImage::from_pixel(width, height, bg_color);

    // Lastly, a scale context.
    let mut scale_cx = ScaleContext::new();

    // Iterate over laid out lines
    for line in layout.lines() {
        // Iterate over GlyphRun's within each line
        for item in line.items() {
            match item {
                PositionedLayoutItem::GlyphRun(glyph_run) => {
                    render_glyph_run(&mut scale_cx, &glyph_run, &mut img, padding);
                }
                PositionedLayoutItem::InlineBox(inline_box) => {
                    for x_off in 0..(inline_box.width.floor() as u32) {
                        for y_off in 0..(inline_box.height.floor() as u32) {
                            let x = inline_box.x as u32 + x_off + padding;
                            let y = inline_box.y as u32 + y_off + padding;
                            img.put_pixel(x, y, Rgba([0, 0, 0, 255]));
                        }
                    }
                }
            }
        }
    }

    // Write image to PNG file in examples/_output dir
    let output_path = {
        let path = std::path::PathBuf::from(file!());
        let mut path = std::fs::canonicalize(path).unwrap();
        path.pop();
        path.pop();
        path.pop();
        path.push("_output");
        drop(std::fs::create_dir(path.clone()));
        path.push("swash_render_font_fallbacks.png");
        path
    };
    let output_file = File::create(output_path).unwrap();
    let png_encoder = PngEncoder::new(output_file);
    img.write_with_encoder(png_encoder).unwrap();
}

/*fn main() {
    // The text we are going to style and lay out
    let text = String::from(
        "Some text here. Let's make it a bit longer so that line wrapping kicks in 😊. And also some اللغة العربية arabic text.\nThis is underline and strikethrough text",
    );

    // The display scale for HiDPI rendering
    let display_scale = 1.0;

    // Whether to automatically align the output to pixel boundaries, to avoid blurry text.
    let quantize = true;

    // The width for line wrapping
    let max_advance = Some(200.0 * display_scale);

    // Colours for rendering
    let text_color = Rgba([0, 0, 0, 255]);
    let bg_color = Rgba([255, 255, 255, 255]);

    // Padding around the output image
    let padding = 20;

    // Create a FontContext, LayoutContext and ScaleContext
    //
    // These are all intended to be constructed rarely (perhaps even once per app (or once per thread))
    // and provide caches and scratch space to avoid allocations
    let mut font_cx = FontContext::new();
    let mut layout_cx = LayoutContext::new();
    let mut scale_cx = ScaleContext::new();

    // Setup some Parley text styles
    let text_brush = ColorBrush { color: text_color };
    let brush_style = StyleProperty::Brush(text_brush);
    let font_stack = FontStack::from("system-ui");
    let bold_style = StyleProperty::FontWeight(FontWeight::new(600.0));
    let underline_style = StyleProperty::Underline(true);
    let strikethrough_style = StyleProperty::Strikethrough(true);

    let mut layout = if std::env::args().any(|arg| arg == "--tree") {
        // TREE BUILDER
        // ============

        // TODO: cleanup API

        let root_style = TextStyle {
            brush: text_brush,
            font_stack,
            line_height: LineHeight::FontSizeRelative(1.3),
            font_size: 16.0,
            ..TextStyle::default()
        };

        let mut builder =
            layout_cx.tree_builder(&mut font_cx, display_scale, quantize, &root_style);

        builder.push_style_modification_span(&[bold_style]);
        builder.push_text(&text[0..5]);
        builder.pop_style_span();

        builder.push_text(&text[5..40]);

        builder.push_inline_box(InlineBox {
            id: 0,
            index: 0,
            width: 50.0,
            height: 50.0,
        });

        builder.push_text(&text[40..50]);

        builder.push_inline_box(InlineBox {
            id: 1,
            index: 50,
            width: 50.0,
            height: 30.0,
        });

        builder.push_text(&text[50..141]);

        // Set the underline style
        builder.push_style_modification_span(&[underline_style]);
        builder.push_text(&text[141..150]);

        builder.pop_style_span();
        builder.push_text(&text[150..155]);

        // Set the strikethrough style
        builder.push_style_modification_span(&[strikethrough_style]);
        builder.push_text(&text[155..168]);

        builder.pop_style_span();
        builder.push_text(&text[168..]);

        // Build the builder into a Layout
        // let mut layout: Layout<ColorBrush> = builder.build(&text);
        let (layout, _text): (Layout<ColorBrush>, String) = builder.build();
        layout
    } else {
        // RANGED BUILDER
        // ============

        // Creates a RangedBuilder
        let mut builder = layout_cx.ranged_builder(&mut font_cx, &text, display_scale, quantize);

        // Set default text colour styles (set foreground text color)
        builder.push_default(brush_style);

        // Set default font family
        builder.push_default(font_stack);
        builder.push_default(LineHeight::FontSizeRelative(1.3));
        builder.push_default(StyleProperty::FontSize(16.0));

        // Set the first 4 characters to bold
        builder.push(bold_style, 0..4);

        // Set the underline & strikethrough style
        builder.push(underline_style, 141..150);
        builder.push(strikethrough_style, 155..168);

        builder.push_inline_box(InlineBox {
            id: 0,
            index: 40,
            width: 50.0,
            height: 50.0,
        });
        builder.push_inline_box(InlineBox {
            id: 1,
            index: 50,
            width: 50.0,
            height: 30.0,
        });

        // Build the builder into a Layout
        // let mut layout: Layout<ColorBrush> = builder.build(&text);
        let layout: Layout<ColorBrush> = builder.build(&text);
        layout
    };

    // Perform layout (including bidi resolution and shaping) with start alignment
    layout.break_all_lines(max_advance);
    layout.align(max_advance, Alignment::Start, AlignmentOptions::default());

    // Create image to render into
    let width = layout.width().ceil() as u32 + (padding * 2);
    let height = layout.height().ceil() as u32 + (padding * 2);
    let mut img = RgbaImage::from_pixel(width, height, bg_color);

    // Iterate over laid out lines
    for line in layout.lines() {
        // Iterate over GlyphRun's within each line
        for item in line.items() {
            match item {
                PositionedLayoutItem::GlyphRun(glyph_run) => {
                    render_glyph_run(&mut scale_cx, &glyph_run, &mut img, padding);
                }
                PositionedLayoutItem::InlineBox(inline_box) => {
                    for x_off in 0..(inline_box.width.floor() as u32) {
                        for y_off in 0..(inline_box.height.floor() as u32) {
                            let x = inline_box.x as u32 + x_off + padding;
                            let y = inline_box.y as u32 + y_off + padding;
                            img.put_pixel(x, y, Rgba([0, 0, 0, 255]));
                        }
                    }
                }
            }
        }
    }

    // Write image to PNG file in examples/_output dir
    let output_path = {
        let path = std::path::PathBuf::from(file!());
        let mut path = std::fs::canonicalize(path).unwrap();
        path.pop();
        path.pop();
        path.pop();
        path.push("_output");
        drop(std::fs::create_dir(path.clone()));
        path.push("swash_render.png");
        path
    };
    let output_file = File::create(output_path).unwrap();
    let png_encoder = PngEncoder::new(output_file);
    img.write_with_encoder(png_encoder).unwrap();
}*/

fn render_glyph_run(
    context: &mut ScaleContext,
    glyph_run: &GlyphRun<'_, ColorBrush>,
    img: &mut RgbaImage,
    padding: u32,
) {
    // Resolve properties of the GlyphRun
    let mut run_x = glyph_run.offset();
    let run_y = glyph_run.baseline();
    let style = glyph_run.style();
    let color = style.brush;

    // Get the "Run" from the "GlyphRun"
    let run = glyph_run.run();

    // Resolve properties of the Run
    let font = run.font();
    let font_size = run.font_size();
    let normalized_coords = run.normalized_coords();

    // Convert from parley::Font to swash::FontRef
    let font_ref = FontRef::from_index(font.data.as_ref(), font.index as usize).unwrap();

    // Build a scaler. As the font properties are constant across an entire run of glyphs
    // we can build one scaler for the run and reuse it for each glyph.
    let mut scaler = context
        .builder(font_ref)
        .size(font_size)
        .hint(true)
        .normalized_coords(normalized_coords)
        .build();

    // Iterates over the glyphs in the GlyphRun
    for glyph in glyph_run.glyphs() {
        let glyph_x = run_x + glyph.x + (padding as f32);
        let glyph_y = run_y - glyph.y + (padding as f32);
        run_x += glyph.advance;

        render_glyph(img, &mut scaler, color, glyph, glyph_x, glyph_y);
    }

    // Draw decorations: underline & strikethrough
    let style = glyph_run.style();
    let run_metrics = run.metrics();
    if let Some(decoration) = &style.underline {
        let offset = decoration.offset.unwrap_or(run_metrics.underline_offset);
        let size = decoration.size.unwrap_or(run_metrics.underline_size);
        render_decoration(img, glyph_run, decoration.brush, offset, size, padding);
    }
    if let Some(decoration) = &style.strikethrough {
        let offset = decoration
            .offset
            .unwrap_or(run_metrics.strikethrough_offset);
        let size = decoration.size.unwrap_or(run_metrics.strikethrough_size);
        render_decoration(img, glyph_run, decoration.brush, offset, size, padding);
    }
}

fn render_decoration(
    img: &mut RgbaImage,
    glyph_run: &GlyphRun<'_, ColorBrush>,
    brush: ColorBrush,
    offset: f32,
    width: f32,
    padding: u32,
) {
    let y = glyph_run.baseline() - offset;
    for pixel_y in y as u32..(y + width) as u32 {
        for pixel_x in glyph_run.offset() as u32..(glyph_run.offset() + glyph_run.advance()) as u32
        {
            img.get_pixel_mut(pixel_x + padding, pixel_y + padding)
                .blend(&brush.color);
        }
    }
}

fn render_glyph(
    img: &mut RgbaImage,
    scaler: &mut Scaler<'_>,
    brush: ColorBrush,
    glyph: Glyph,
    glyph_x: f32,
    glyph_y: f32,
) {
    // Compute the fractional offset
    // You'll likely want to quantize this in a real renderer
    let offset = Vector::new(glyph_x.fract(), glyph_y.fract());

    // Render the glyph using swash
    let rendered_glyph = Render::new(
        // Select our source order
        &[
            Source::ColorOutline(0),
            Source::ColorBitmap(StrikeWith::BestFit),
            Source::Outline,
        ],
    )
    // Select the simple alpha (non-subpixel) format
    .format(Format::Alpha)
    // Apply the fractional offset
    .offset(offset)
    // Render the image
    .render(scaler, glyph.id as u16)
    .unwrap();

    let glyph_width = rendered_glyph.placement.width;
    let glyph_height = rendered_glyph.placement.height;
    let glyph_x = (glyph_x.floor() as i32 + rendered_glyph.placement.left) as u32;
    let glyph_y = (glyph_y.floor() as i32 - rendered_glyph.placement.top) as u32;

    match rendered_glyph.content {
        Content::Mask => {
            let mut i = 0;
            let bc = brush.color;
            for pixel_y in 0..glyph_height {
                for pixel_x in 0..glyph_width {
                    let x = glyph_x + pixel_x;
                    let y = glyph_y + pixel_y;
                    let alpha = rendered_glyph.data[i];
                    let color = Rgba([bc[0], bc[1], bc[2], alpha]);
                    img.get_pixel_mut(x, y).blend(&color);
                    i += 1;
                }
            }
        }
        Content::SubpixelMask => unimplemented!(),
        Content::Color => {
            let row_size = glyph_width as usize * 4;
            for (pixel_y, row) in rendered_glyph.data.chunks_exact(row_size).enumerate() {
                for (pixel_x, pixel) in row.chunks_exact(4).enumerate() {
                    let x = glyph_x + pixel_x as u32;
                    let y = glyph_y + pixel_y as u32;
                    let color = Rgba(pixel.try_into().expect("Not RGBA"));
                    img.get_pixel_mut(x, y).blend(&color);
                }
            }
        }
    }
}
