// Colormaps. Each sample() writes [r,g,b] in 0..1 into out and returns out.

export type RGB = [number, number, number];

function lerpStops(stops: RGB[], t: number, out: RGB): RGB {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  out[0] = stops[i][0] + (stops[i + 1][0] - stops[i][0]) * f;
  out[1] = stops[i][1] + (stops[i + 1][1] - stops[i][1]) * f;
  out[2] = stops[i][2] + (stops[i + 1][2] - stops[i][2]) * f;
  return out;
}

/** 5-stop plasma (port of pc_viewer.html PLASMA) */
const PLASMA_STOPS: RGB[] = [
  [13 / 255, 8 / 255, 129 / 255],
  [126 / 255, 3 / 255, 168 / 255],
  [204 / 255, 71 / 255, 120 / 255],
  [248 / 255, 149 / 255, 64 / 255],
  [240 / 255, 249 / 255, 33 / 255],
];

export function plasma(t: number, out: RGB = [0, 0, 0]): RGB {
  return lerpStops(PLASMA_STOPS, t, out);
}

/** diverging blue->red (port of pc_viewer.html RDBU): blue = nearer/smaller, red = farther */
const RDBU_STOPS: RGB[] = [
  [5 / 255, 113 / 255, 176 / 255],
  [67 / 255, 147 / 255, 195 / 255],
  [146 / 255, 197 / 255, 222 / 255],
  [247 / 255, 247 / 255, 247 / 255],
  [244 / 255, 165 / 255, 130 / 255],
  [214 / 255, 96 / 255, 77 / 255],
  [103 / 255, 0, 31 / 255],
];

export function rdbu(t: number, out: RGB = [0, 0, 0]): RGB {
  return lerpStops(RDBU_STOPS, t, out);
}

/** Google turbo colormap (polynomial approximation) */
export function turbo(t: number, out: RGB = [0, 0, 0]): RGB {
  const x = Math.min(1, Math.max(0, t));
  out[0] = 0.1357 + x * (4.5974 - x * (42.3277 - x * (130.7177 - x * (150.2865 - x * 58.2358))));
  out[1] = 0.0914 + x * (2.1856 + x * (4.8052 - x * (14.0195 - x * (10.8986 - x * 1.9))));
  out[2] = 0.1067 + x * (14.1616 - x * (37.5938 - x * (28.7252 - x * (3.9904 + x * 0.3))));
  out[0] = Math.min(1, Math.max(0, out[0]));
  out[1] = Math.min(1, Math.max(0, out[1]));
  out[2] = Math.min(1, Math.max(0, out[2]));
  return out;
}

/** classic jet */
export function jet(t: number, out: RGB = [0, 0, 0]): RGB {
  const x = Math.min(1, Math.max(0, t)) * 1.5; // map to 0..1.5 segment style
  const r = Math.min(1, Math.max(0, 1.5 - Math.abs(2 * x - 1.5)));
  const g = Math.min(1, Math.max(0, 1.5 - Math.abs(2 * x - 1.0)));
  const b = Math.min(1, Math.max(0, 1.5 - Math.abs(2 * x - 0.5)));
  out[0] = r; out[1] = g; out[2] = b;
  return out;
}

export type ColormapName = "plasma" | "turbo" | "jet";

export function colormap(name: ColormapName, t: number, out: RGB = [0, 0, 0]): RGB {
  switch (name) {
    case "turbo":
      return turbo(t, out);
    case "jet":
      return jet(t, out);
    default:
      return plasma(t, out);
  }
}
