//! Draw ink as SVG, so a fit can be looked at rather than only measured.
//!
//! A CTC loss going down says the optimizer worked; it does not say whether the
//! result still looks like handwriting. That is a question only an eye can
//! answer, which is what this is for.

use std::fmt::Write;

use mlkit_ink::Stroke;

/// Overlay two inks: the original faint, the result solid, with a light
/// connector at each moved sample so the displacement field is visible.
pub fn overlay(before: &[Stroke], after: &[Stroke], caption: &str) -> String {
    let (min, max) = bounds(before.iter().chain(after));
    let pad = 0.08 * (max.0 - min.0).max(max.1 - min.1).max(1.0);
    let (width, height) = (max.0 - min.0 + 2.0 * pad, max.1 - min.1 + 2.0 * pad);
    let stroke_width = 0.012 * width.max(height);

    let mut out = String::new();
    let _ = write!(
        out,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{:.3} {:.3} {:.3} {:.3}" width="480">"#,
        min.0 - pad,
        min.1 - pad,
        width,
        height
    );
    let _ = write!(
        out,
        r##"<rect x="{:.3}" y="{:.3}" width="{width:.3}" height="{height:.3}" fill="#fdfdfb"/>"##,
        min.0 - pad,
        min.1 - pad
    );

    for (a, b) in before.iter().zip(after) {
        for i in 0..a.x.len().min(b.x.len()) {
            let _ = write!(
                out,
                r##"<line x1="{:.3}" y1="{:.3}" x2="{:.3}" y2="{:.3}" stroke="#d64545" stroke-width="{:.4}" opacity="0.45"/>"##,
                a.x[i],
                a.y[i],
                b.x[i],
                b.y[i],
                stroke_width * 0.25
            );
        }
    }
    write_strokes(&mut out, before, "#9aa0a6", stroke_width, 0.55);
    write_strokes(&mut out, after, "#1a1a1a", stroke_width, 1.0);

    let _ = write!(
        out,
        r##"<text x="{:.3}" y="{:.3}" font-family="ui-monospace,monospace" font-size="{:.3}" fill="#555">{}</text>"##,
        min.0 - pad + 0.02 * width,
        max.1 + pad - 0.02 * height,
        0.055 * height.max(width),
        escape(caption)
    );
    out.push_str("</svg>\n");
    out
}

fn write_strokes(out: &mut String, strokes: &[Stroke], color: &str, width: f64, opacity: f64) {
    for stroke in strokes {
        if stroke.x.len() < 2 {
            continue;
        }
        let points: Vec<String> = stroke
            .x
            .iter()
            .zip(&stroke.y)
            .map(|(x, y)| format!("{x:.3},{y:.3}"))
            .collect();
        let _ = write!(
            out,
            r#"<polyline points="{}" fill="none" stroke="{color}" stroke-width="{width:.4}" stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"/>"#,
            points.join(" ")
        );
    }
}

fn bounds<'a>(strokes: impl Iterator<Item = &'a Stroke>) -> ((f64, f64), (f64, f64)) {
    let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
    let mut seen = false;
    for stroke in strokes {
        for (&x, &y) in stroke.x.iter().zip(&stroke.y) {
            lo = (lo.0.min(x), lo.1.min(y));
            hi = (hi.0.max(x), hi.1.max(y));
            seen = true;
        }
    }
    if seen {
        (lo, hi)
    } else {
        ((0.0, 0.0), (1.0, 1.0))
    }
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
