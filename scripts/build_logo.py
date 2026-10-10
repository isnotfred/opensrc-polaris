"""Generate the official vector SVG brand logo for Polaris and export multi-resolution icons."""
from pathlib import Path
from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication
import sys

# 512x512 Pure Geometric Vector SVG for Polaris
POLARIS_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <!-- Background Gradient: Deep Obsidian Slate -->
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0b101b" />
      <stop offset="50%" stop-color="#0d1424" />
      <stop offset="100%" stop-color="#070a12" />
    </linearGradient>

    <!-- Subtle Rim Glow -->
    <linearGradient id="borderGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.35" />
      <stop offset="50%" stop-color="#1e293b" stop-opacity="0.2" />
      <stop offset="100%" stop-color="#2563eb" stop-opacity="0.35" />
    </linearGradient>

    <!-- Primary Star Facet Gradients -->
    <linearGradient id="facetNorthWest" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#7dd3fc" />
      <stop offset="100%" stop-color="#38bdf8" />
    </linearGradient>

    <linearGradient id="facetNorthEast" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8" />
      <stop offset="100%" stop-color="#2563eb" />
    </linearGradient>

    <linearGradient id="facetEastNorth" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#60a5fa" />
      <stop offset="100%" stop-color="#2563eb" />
    </linearGradient>

    <linearGradient id="facetEastSouth" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1d4ed8" />
      <stop offset="100%" stop-color="#1e3a8a" />
    </linearGradient>

    <linearGradient id="facetSouthEast" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#2563eb" />
      <stop offset="100%" stop-color="#1d4ed8" />
    </linearGradient>

    <linearGradient id="facetSouthWest" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0284c7" />
      <stop offset="100%" stop-color="#0369a1" />
    </linearGradient>

    <linearGradient id="facetWestSouth" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0284c7" />
      <stop offset="100%" stop-color="#0ea5e9" />
    </linearGradient>

    <linearGradient id="facetWestNorth" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8" />
      <stop offset="100%" stop-color="#7dd3fc" />
    </linearGradient>
  </defs>

  <!-- Squircle Canvas (Apple macOS / Modern Desktop Ratio) -->
  <rect x="16" y="16" width="480" height="480" rx="112" fill="url(#bgGrad)" stroke="url(#borderGrad)" stroke-width="2.5" />

  <!-- Subtle Navigational Orbital Ring -->
  <circle cx="256" cy="256" r="168" fill="none" stroke="#38bdf8" stroke-opacity="0.16" stroke-width="2" stroke-dasharray="8 6" />
  <circle cx="256" cy="256" r="198" fill="none" stroke="#2563eb" stroke-opacity="0.10" stroke-width="1.5" />

  <!-- Secondary 4-Point Star (45° Rotated Compass Rays) -->
  <g opacity="0.85">
    <!-- NW Ray -->
    <polygon points="256,256 226,226 148,148 256,256" fill="#1e293b" />
    <polygon points="256,256 148,148 226,226 256,256" fill="#334155" opacity="0.6" />

    <!-- NE Ray -->
    <polygon points="256,256 286,226 364,148 256,256" fill="#1e293b" />
    <polygon points="256,256 364,148 286,226 256,256" fill="#334155" opacity="0.6" />

    <!-- SE Ray -->
    <polygon points="256,256 286,286 364,364 256,256" fill="#1e293b" />
    <polygon points="256,256 364,364 286,286 256,256" fill="#0f172a" opacity="0.6" />

    <!-- SW Ray -->
    <polygon points="256,256 226,286 148,364 256,256" fill="#1e293b" />
    <polygon points="256,256 148,364 226,286 256,256" fill="#0f172a" opacity="0.6" />
  </g>

  <!-- Primary 4-Point Faceted Polaris Star -->
  <g id="polaris-main-star">
    <!-- NORTH POINT -->
    <!-- Left (West) Half -->
    <polygon points="256,72 256,256 206,206" fill="url(#facetNorthWest)" />
    <!-- Right (East) Half -->
    <polygon points="256,72 306,206 256,256" fill="url(#facetNorthEast)" />

    <!-- EAST POINT -->
    <!-- Top (North) Half -->
    <polygon points="440,256 256,256 306,206" fill="url(#facetEastNorth)" />
    <!-- Bottom (South) Half -->
    <polygon points="440,256 306,306 256,256" fill="url(#facetEastSouth)" />

    <!-- SOUTH POINT -->
    <!-- Right (East) Half -->
    <polygon points="256,440 256,256 306,306" fill="url(#facetSouthEast)" />
    <!-- Left (West) Half -->
    <polygon points="256,440 206,306 256,256" fill="url(#facetSouthWest)" />

    <!-- WEST POINT -->
    <!-- Bottom (South) Half -->
    <polygon points="72,256 256,256 206,306" fill="url(#facetWestSouth)" />
    <!-- Top (North) Half -->
    <polygon points="72,256 206,206 256,256" fill="url(#facetWestNorth)" />
  </g>

  <!-- Center Luminous Diamond Core -->
  <g id="polaris-nucleus">
    <!-- Left Diamond Half (Pure White) -->
    <polygon points="256,224 224,256 256,288" fill="#ffffff" />
    <!-- Right Diamond Half (Ice Cyan Highlight) -->
    <polygon points="256,224 288,256 256,288" fill="#bae6fd" />
    <!-- Center Point Specular Dot -->
    <circle cx="256" cy="256" r="4.5" fill="#ffffff" />
  </g>
</svg>
"""

def main():
    assets_dir = Path("src/polaris/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    svg_path = assets_dir / "icon.svg"
    svg_path.write_text(POLARIS_SVG, encoding="utf-8")
    print(f"Saved vector SVG: {svg_path}")

    # Render multi-resolution PNGs via QtSvg
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    renderer = QSvgRenderer(QByteArray(POLARIS_SVG.encode("utf-8")))

    # 512x512 Master PNG
    img512 = QImage(512, 512, QImage.Format.Format_ARGB32_Premultiplied)
    img512.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img512)
    renderer.render(painter)
    painter.end()

    png_path = assets_dir / "icon.png"
    img512.save(str(png_path), "PNG")
    print(f"Rendered master icon: {png_path} (512x512)")

    # 64x64 and 32x32 for high-DPI and desktop
    img64 = img512.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    img64.save(str(assets_dir / "icon-64.png"), "PNG")

    img32 = img512.scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    img32.save(str(assets_dir / "icon-32.png"), "PNG")

    img16 = img512.scaled(16, 16, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    img16.save(str(assets_dir / "icon-16.png"), "PNG")
    print("Multi-resolution icons (16px, 32px, 64px, 512px) generated successfully!")

if __name__ == "__main__":
    main()
