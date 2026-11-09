"""
GTA Texture Export - Substance 3D Painter plugin
=================================================

Exports GTA V textures for each texture set:

    <name>_d   diffuse   (base color + opacity, metal optionally darkened)
    <name>_n   normal    (DirectX or OpenGL)
    <name>_s   specular  (built from metallic / roughness, see below)

as DDS through NVIDIA Texture Tools (nvtt_export.exe), or as PNG / TGA.
Each texture set gets its own output name and its own power-of-2
resolution per map ('Off' skips that map).

The _s map:
    R = specular intensity   (from roughness + metallic)
    G = specular falloff     (from roughness, 1 = sharp)
    B = specular fresnel     (from metallic)
Every texture set gets a spec *profile* (vehicle paint, chrome, rubber ...)
holding the intensity / fresnel / falloff values for that kind of shader.
Profiles are editable in the panel; assignments are stored in the .spp.

Install
-------
Copy this file to  <Painter user folder>/python/plugins/
then in Painter: Python > Reload Plugins Folder, and tick "gta_spec_export".
DDS export needs the NVIDIA Texture Tools Exporter (standalone) installed.
"""

import json
import os
import shutil
import subprocess
import tempfile
import traceback

import substance_painter.event as sp_event
import substance_painter.export as sp_export
import substance_painter.logging as sp_log
import substance_painter.project as sp_project
import substance_painter.textureset as sp_ts
import substance_painter.ui as sp_ui

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:  # Painter versions before 10.1
    from PySide2 import QtCore, QtGui, QtWidgets

try:
    import numpy as np
except Exception:  # numpy is optional, only used to go faster
    np = None


PLUGIN_TITLE = "GTA Texture Export"
METADATA_CONTEXT = "gta_spec_export"
PROFILES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "gta_spec_export_profiles.json")

# Used when a texture set has no roughness / metallic channel
DEFAULT_ROUGHNESS = 0.5
DEFAULT_METALLIC = 0.0


# ============================================================== profiles ===
# Starting points only - tune them against the game.
DEFAULT_PROFILES = [
    {"name": "Generic (vehicle_generic)",
     "intensity_smooth": 0.6, "intensity_rough": 0.05, "intensity_metal": 0.78,
     "shader_intensity_mult": 1.0,
     "fresnel_dielectric": 0.6, "fresnel_metal": 0.93,
     "falloff_mode": "linear", "falloff_max": 0.75, "falloff_min": 0.0,
     "falloff_mult": 512.0},
    {"name": "Vehicle paint",
     "intensity_smooth": 1.0, "intensity_rough": 0.3, "intensity_metal": 1.0,
     "shader_intensity_mult": 1.0,
     "fresnel_dielectric": 1.0, "fresnel_metal": 1.0,
     "falloff_mode": "linear", "falloff_max": 1.0, "falloff_min": 0.0,
     "falloff_mult": 512.0},
    {"name": "Chrome / bare metal",
     "intensity_smooth": 0.78, "intensity_rough": 0.4, "intensity_metal": 0.78,
     "shader_intensity_mult": 1.0,
     "fresnel_dielectric": 0.93, "fresnel_metal": 0.93,
     "falloff_mode": "linear", "falloff_max": 0.75, "falloff_min": 0.0,
     "falloff_mult": 512.0},
    {"name": "Rubber / tyre",
     "intensity_smooth": 0.25, "intensity_rough": 0.0, "intensity_metal": 0.25,
     "shader_intensity_mult": 1.0,
     "fresnel_dielectric": 0.3, "fresnel_metal": 0.3,
     "falloff_mode": "linear", "falloff_max": 0.5, "falloff_min": 0.0,
     "falloff_mult": 512.0},
    {"name": "Glass",
     "intensity_smooth": 1.0, "intensity_rough": 1.0, "intensity_metal": 1.0,
     "shader_intensity_mult": 1.0,
     "fresnel_dielectric": 1.0, "fresnel_metal": 1.0,
     "falloff_mode": "linear", "falloff_max": 1.0, "falloff_min": 0.0,
     "falloff_mult": 512.0},
]

# Values the built-in profiles had in version 1 (upgraded automatically if untouched)
_V1_FRESNEL_METAL = {"Generic (vehicle_generic)": 0.2, "Vehicle paint": 0.5,
                     "Chrome / bare metal": 0.2, "Rubber / tyre": 1.0, "Glass": 1.0}

# key, label, min, max, step, tooltip
PROFILE_FIELDS = [
    ("intensity_smooth", "Intensity (smooth)", 0.0, 1.0, 0.05,
     "R for a fully smooth non-metal (roughness 0)"),
    ("intensity_rough", "Intensity (rough)", 0.0, 1.0, 0.05,
     "R for a fully rough non-metal (roughness 1). Same as 'smooth' = constant intensity"),
    ("intensity_metal", "Intensity (metal)", 0.0, 1.0, 0.05,
     "R where metallic = 1"),
    ("shader_intensity_mult", "Shader specIntensityMult", 0.01, 10.0, 0.1,
     "Divides R by this value. Leave at 1 - Rockstar's own maps are authored "
     "without compensating for the shader's matDiffuseColor"),
    ("fresnel_dielectric", "Fresnel (non-metal)", 0.0, 1.0, 0.05,
     "B for non-metal areas"),
    ("fresnel_metal", "Fresnel (metal)", 0.0, 1.0, 0.05,
     "B where metallic = 1"),
    ("falloff_max", "Falloff at roughness 0", 0.0, 1.0, 0.05,
     "G for a perfectly smooth surface (Rockstar chrome is about 0.75)"),
    ("falloff_min", "Falloff at roughness 1", 0.0, 1.0, 0.05,
     "G for a fully rough surface"),
    ("falloff_mult", "Shader specFalloffMult", 1.0, 4096.0, 16.0,
     "Only used in Phong mode: G = Blinn-Phong exponent / this value"),
]


def load_profiles():
    try:
        with open(PROFILES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):              # version 1 file
            data = _migrate_v1(data)
        elif isinstance(data, dict):
            data = data.get("profiles", [])
        if data:
            out = []
            for p in data:
                full = dict(DEFAULT_PROFILES[0])
                full.update(p)
                out.append(full)
            return out
    except (OSError, ValueError):
        pass
    return [dict(p) for p in DEFAULT_PROFILES]


def _migrate_v1(profiles):
    """Replace built-in profiles that still have the old (wrong) metal fresnel."""
    builtin = {d["name"]: d for d in DEFAULT_PROFILES}
    out = []
    for p in profiles:
        name = p.get("name")
        if name in builtin and p.get("fresnel_metal") == _V1_FRESNEL_METAL.get(name):
            out.append(dict(builtin[name]))
        else:
            out.append(p)
    return out


def save_profiles(profiles):
    try:
        with open(PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump({"version": 2, "profiles": profiles}, f, indent=2)
    except OSError as e:
        sp_log.warning(f"{PLUGIN_TITLE}: could not save profiles: {e}")


# ========================================================== pixel maths ===
def _clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _to8(x):
    return int(round(_clamp01(x) * 255.0))


def roughness_to_falloff(r, prof):
    if prof["falloff_mode"] == "phong":
        # perceptual roughness -> GGX alpha -> Blinn-Phong exponent -> / mult
        alpha = max(r * r, 1e-4)
        n = 2.0 / (alpha * alpha) - 2.0
        g = _clamp01(n / max(prof["falloff_mult"], 1e-4))
    else:
        g = 1.0 - r
    return prof["falloff_min"] + (prof["falloff_max"] - prof["falloff_min"]) * g


def spec_intensity(r, m, prof):
    i = prof["intensity_rough"] + (prof["intensity_smooth"] - prof["intensity_rough"]) * (1.0 - r)
    i = i + (prof["intensity_metal"] - i) * m
    return i / max(prof["shader_intensity_mult"], 1e-4)


def spec_fresnel(m, prof):
    return prof["fresnel_dielectric"] + (prof["fresnel_metal"] - prof["fresnel_dielectric"]) * m


def lut1d(fn):
    return bytes(_to8(fn(i / 255.0)) for i in range(256))


def lut2d(fn):
    """65536 entry table, index = a * 256 + b."""
    return bytes(_to8(fn(a / 255.0, b / 255.0)) for a in range(256) for b in range(256))


def apply2d(lut, a, b):
    """Per-pixel lut[a * 256 + b] for two 8-bit channel byte strings."""
    n = len(a)
    if b[:1] * n == b:                     # b is uniform -> a plain 1D lookup
        k = b[0]
        return a.translate(bytes(lut[x * 256 + k] for x in range(256)))
    if np is not None:
        ia = np.frombuffer(a, np.uint8).astype(np.int32)
        ib = np.frombuffer(b, np.uint8).astype(np.int32)
        return np.frombuffer(lut, np.uint8)[ia * 256 + ib].tobytes()
    return bytes(lut[x * 256 + y] for x, y in zip(a, b))


def interleave(r, g, b, a):
    out = bytearray(len(r) * 4)
    out[0::4], out[1::4], out[2::4], out[3::4] = r, g, b, a
    return bytes(out)


def build_spec(rough, metal, prof):
    """rough / metal: 8-bit single channel byte strings of equal length."""
    n = len(rough)
    r = apply2d(lut2d(lambda ro, me: spec_intensity(ro, me, prof)), rough, metal)
    g = rough.translate(lut1d(lambda ro: roughness_to_falloff(ro, prof)))
    b = metal.translate(lut1d(lambda me: spec_fresnel(me, prof)))
    return interleave(r, g, b, b"\xff" * n)


def build_diffuse(base_rgba, metal, darken):
    """Darken base color where metallic, the way GTA chrome is authored."""
    lut = lut2d(lambda c, m: c * (1.0 - m * darken))
    out = bytearray(base_rgba)
    for i in range(3):
        out[i::4] = apply2d(lut, base_rgba[i::4], metal)
    return bytes(out)


# ============================================================ image I/O ===
_RGBA8888 = getattr(QtGui.QImage, "Format_RGBA8888", None)
if _RGBA8888 is None:
    _RGBA8888 = QtGui.QImage.Format.Format_RGBA8888


def read_rgba(path):
    img = QtGui.QImage(path)
    if img.isNull():
        raise RuntimeError(f"Could not read {path}")
    img = img.convertToFormat(_RGBA8888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    ptr = img.constBits()
    if hasattr(ptr, "setsize"):
        ptr.setsize(img.sizeInBytes())
    data = bytes(ptr)
    if bpl != w * 4:
        data = b"".join(data[y * bpl:y * bpl + w * 4] for y in range(h))
    return w, h, data


def write_rgba(path, w, h, data):
    img = QtGui.QImage(data, w, h, w * 4, _RGBA8888).copy()
    if not img.save(path):
        raise RuntimeError(f"Could not write {path}")


def resize_rgba(w, h, data, tw, th):
    if (w, h) == (tw, th):
        return data
    img = QtGui.QImage(data, w, h, w * 4, _RGBA8888).copy()
    img = img.scaled(tw, th, QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                     QtCore.Qt.TransformationMode.SmoothTransformation)
    return read_qimage(img)[2]


def read_qimage(img):
    img = img.convertToFormat(_RGBA8888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    ptr = img.constBits()
    if hasattr(ptr, "setsize"):
        ptr.setsize(img.sizeInBytes())
    data = bytes(ptr)
    if bpl != w * 4:
        data = b"".join(data[y * bpl:y * bpl + w * 4] for y in range(h))
    return w, h, data


# ================================================================ DDS ===
DEFAULT_NVTT = r"C:\Program Files\NVIDIA Corporation\NVIDIA Texture Tools\nvtt_export.exe"

# label, nvtt format ("auto" = bc1, or bc3 when the image has alpha)
D_FORMATS = [("Auto: DXT1, DXT5 if alpha", "auto"), ("BC1 / DXT1", "bc1"),
             ("BC3 / DXT5", "bc3"), ("BC7", "bc7"), ("BGRA8 (uncompressed)", "bgra8")]
N_FORMATS = [("BC1 / DXT1", "bc1"), ("BC3 / DXT5", "bc3"), ("BC5 / ATI2", "bc5"),
             ("BC7", "bc7"), ("BGRA8 (uncompressed)", "bgra8")]
S_FORMATS = [("BC1 / DXT1", "bc1"), ("BC3 / DXT5", "bc3"), ("BC7", "bc7"),
             ("BGRA8 (uncompressed)", "bgra8")]
QUALITIES = ["fastest", "normal", "production", "highest"]
RESOLUTIONS = ["Off", "128", "256", "512", "1024", "2048", "4096", "8192"]


def run_nvtt(nvtt, src, dst, fmt, quality, mips, extra, log):
    cmd = [nvtt, src, "--format", fmt, "--quality", quality, "--output", dst]
    if not mips:
        cmd.append("--no-mips")
    if extra.strip():
        cmd += extra.split()
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    res = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags)
    if res.returncode != 0 or not os.path.exists(dst):
        msg = (res.stderr or res.stdout or "").strip().splitlines()
        raise RuntimeError(f"NVTT failed on {os.path.basename(dst)}: "
                           f"{msg[-1] if msg else 'exit code ' + str(res.returncode)}")


# ============================================================== export ===
MAP_PARAMS = {"fileFormat": "png", "bitDepth": "8", "dithering": False,
              "paddingAlgorithm": "infinite"}
T_D_BASE, T_D_METAL = "__gt_d_base", "__gt_d_metal"
T_N = "__gt_n"
T_S_ROUGH, T_S_METAL = "__gt_s_rough", "__gt_s_metal"


def _params(res):
    p = dict(MAP_PARAMS)
    p["sizeLog2"] = int(res).bit_length() - 1
    return p


def _gray_map(suffix, src, res):
    return {"fileName": "$textureSet" + suffix,
            "channels": [{"destChannel": "L", "srcChannel": "L",
                          "srcMapType": "documentMap", "srcMapName": src}],
            "parameters": _params(res)}


def _rgb_map(suffix, src_type, src_name, res, opacity=False):
    chans = [{"destChannel": c, "srcChannel": c, "srcMapType": src_type,
              "srcMapName": src_name} for c in "RGB"]
    if opacity:
        chans.append({"destChannel": "A", "srcChannel": "L",
                      "srcMapType": "documentMap", "srcMapName": "opacity"})
    return {"fileName": "$textureSet" + suffix, "channels": chans,
            "parameters": _params(res)}


def export_gta(jobs, opts, log=print):
    """
    jobs: list of dicts {ts, name, profile, res: {"d": int|None, "n": .., "s": ..}}
    opts: out_dir, filetype ("dds"/"png"/"tga"), fmt_d, fmt_n, fmt_s, normal ("DirectX"/"OpenGL"),
          darken (float, 0 = off), mips, quality, nvtt, extra
    """
    out_dir = opts["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    ext = "." + opts["filetype"]
    if opts["filetype"] == "dds" and not os.path.isfile(opts["nvtt"]):
        raise RuntimeError(f"nvtt_export.exe not found at: {opts['nvtt']}")

    CT = sp_ts.ChannelType
    tmp = tempfile.mkdtemp(prefix="gtatex_")
    written = []
    try:
        presets, export_list, plans = [], [], []
        for i, job in enumerate(jobs):
            ts = sp_ts.TextureSet.from_name(job["ts"])
            stack = ts.get_stack()
            has = {ct: stack.has_channel(ct) for ct in
                   (CT.Roughness, CT.Metallic, CT.BaseColor, CT.Opacity, CT.Normal, CT.Height)}
            res = job["res"]
            maps = []
            plan = dict(job, has=has, out=[])

            if res["d"]:
                if has[CT.BaseColor]:
                    maps.append(_rgb_map(T_D_BASE, "documentMap", "basecolor", res["d"],
                                         opacity=has[CT.Opacity]))
                    if opts["darken"] > 0 and has[CT.Metallic]:
                        maps.append(_gray_map(T_D_METAL, "metallic", res["d"]))
                    plan["out"].append("d")
                else:
                    log(f"[skip] {job['ts']} _d: no Base Color channel")
            if res["n"]:
                maps.append(_rgb_map(T_N, "virtualMap", "Normal_" + opts["normal"], res["n"]))
                plan["out"].append("n")
            if res["s"]:
                if has[CT.Roughness] or has[CT.Metallic]:
                    if has[CT.Roughness]:
                        maps.append(_gray_map(T_S_ROUGH, "roughness", res["s"]))
                    if has[CT.Metallic]:
                        maps.append(_gray_map(T_S_METAL, "metallic", res["s"]))
                    plan["out"].append("s")
                else:
                    log(f"[skip] {job['ts']} _s: no roughness or metallic channel")
            if not maps:
                continue
            pname = f"gtatex_tmp_{i}"
            presets.append({"name": pname, "maps": maps})
            export_list.append({"rootPath": job["ts"], "exportPreset": pname})
            plans.append(plan)

        if not plans:
            log("Nothing to export.")
            return written

        log("Rendering maps in Painter...")
        result = sp_export.export_project_textures({
            "exportShaderParams": False,
            "exportPath": tmp,
            "defaultExportPreset": presets[0]["name"],
            "exportPresets": presets,
            "exportList": export_list,
        })
        if result.status != sp_export.ExportStatus.Success:
            raise RuntimeError(f"Painter export failed: {result.message}")

        files = {}
        for paths in result.textures.values():
            for p in paths:
                files[os.path.splitext(os.path.basename(p))[0]] = p

        def load(ts_name, suffix, size):
            p = files.get(ts_name + suffix) or os.path.join(tmp, ts_name + suffix + ".png")
            if not os.path.exists(p):
                return None
            w, h, data = read_rgba(p)
            if size and (w, h) != size:          # in case Painter ignored sizeLog2
                data = resize_rgba(w, h, data, *size)
            return data

        def target_size(ts_name, res):
            r = sp_ts.TextureSet.from_name(ts_name).get_resolution()
            longest = max(r.width, r.height)
            return (max(1, r.width * res // longest), max(1, r.height * res // longest))

        def finish(job, kind, size, rgba, fmt):
            """Write the final file, via NVTT for DDS."""
            final = os.path.join(out_dir, job["files"][kind] + ext)
            if opts["filetype"] == "dds":
                if fmt == "auto":
                    fmt = "bc1" if rgba[3::4].count(b"\xff") == size[0] * size[1] else "bc3"
                src = os.path.join(tmp, f"final_{len(written)}_{kind}.png")
                write_rgba(src, size[0], size[1], rgba)
                run_nvtt(opts["nvtt"], src, final, fmt, opts["quality"], opts["mips"],
                         opts["extra"], log)
                label = fmt.upper()
            else:
                write_rgba(final, size[0], size[1], rgba)
                label = opts["filetype"].upper()
            written.append(final)
            log(f"[ok] {os.path.basename(final)}  {size[0]}x{size[1]}  {label}")

        for plan in plans:
            ts_name, has = plan["ts"], plan["has"]
            try:
                if "d" in plan["out"]:
                    size = target_size(ts_name, plan["res"]["d"])
                    base = load(ts_name, T_D_BASE, size)
                    if base is None:
                        log(f"[fail] {ts_name} _d: Painter did not export base color")
                    else:
                        if opts["darken"] > 0 and has[CT.Metallic]:
                            metal = load(ts_name, T_D_METAL, size)
                            if metal is not None:
                                base = build_diffuse(base, metal[0::4], opts["darken"])
                        finish(plan, "d", size, base, opts["fmt_d"])

                if "n" in plan["out"]:
                    size = target_size(ts_name, plan["res"]["n"])
                    nrm = load(ts_name, T_N, size)
                    if nrm is None:
                        log(f"[fail] {ts_name} _n: Painter did not export the normal map")
                    else:
                        finish(plan, "n", size, nrm, opts["fmt_n"])

                if "s" in plan["out"]:
                    size = target_size(ts_name, plan["res"]["s"])
                    n = size[0] * size[1]
                    rough = load(ts_name, T_S_ROUGH, size) if has[CT.Roughness] else None
                    metal = load(ts_name, T_S_METAL, size) if has[CT.Metallic] else None
                    rough = rough[0::4] if rough else bytes([_to8(DEFAULT_ROUGHNESS)]) * n
                    metal = metal[0::4] if metal else bytes([_to8(DEFAULT_METALLIC)]) * n
                    finish(plan, "s", size, build_spec(rough, metal, plan["profile"]),
                           opts["fmt_s"])
            except RuntimeError as e:
                log(f"[fail] {ts_name}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return written


# ================================================================== UI ===
COL_TS, COL_NAME, COL_MAPS = range(3)
MAP_KEYS = ("d", "n", "s")
MAP_LABELS = {"d": "Diffuse _d", "n": "Normal _n", "s": "Specular _s"}
RES_SIZES = RESOLUTIONS[1:]


class GTASpecWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("GTASpecExport")
        self.setWindowTitle(PLUGIN_TITLE)
        self.settings = QtCore.QSettings("GTATools", "GTASpecExport")
        self.profiles = load_profiles()
        self._loading = False
        self.rows, self.cur_ts = {}, None
        self._loading = True          # nothing may be saved before the rows are loaded
        self._build_ui()
        self._loading = False
        self._load_profile_into_form()

        self._events = [
            (sp_event.ProjectEditionEntered, self._on_project_open),
            (sp_event.ProjectAboutToClose, self._on_project_close),
        ]
        for ev, cb in self._events:
            sp_event.DISPATCHER.connect(ev, cb)
        if sp_project.is_open():
            self.refresh_texture_sets()

    # ------------------------------------------------------------ build ---
    def _combo(self, items, key, default):
        c = QtWidgets.QComboBox()
        for label, data in items:
            c.addItem(label, data)
        i = c.findData(self.settings.value(key, default))
        c.setCurrentIndex(max(i, 0))
        return c

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        body = QtWidgets.QWidget()
        scroll.setWidget(body)
        lay = QtWidgets.QVBoxLayout(body)

        # ---- texture sets
        grp = QtWidgets.QGroupBox("Texture sets")
        gl = QtWidgets.QVBoxLayout(grp)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Texture set", "Export name", "Maps"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setMinimumHeight(120)
        self.table.itemChanged.connect(self._table_item_changed)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        gl.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Refresh")
        b.clicked.connect(self.refresh_texture_sets)
        row.addWidget(b)
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("All resolutions:"))
        self.res_all = QtWidgets.QComboBox()
        self.res_all.addItems(RES_SIZES)
        self.res_all.setCurrentText("2048")
        row.addWidget(self.res_all)
        b = QtWidgets.QPushButton("Apply")
        b.clicked.connect(self._apply_res_all)
        row.addWidget(b)
        gl.addLayout(row)
        lay.addWidget(grp)

        # ---- selected texture set
        self.detail = QtWidgets.QGroupBox("Selected texture set")
        form = QtWidgets.QFormLayout(self.detail)
        self.d_name = QtWidgets.QLineEdit()
        self.d_name.setToolTip("Base name - maps become <name>_d, <name>_n, <name>_s")
        self.d_name.textEdited.connect(self._detail_changed)
        form.addRow("Export name", self.d_name)
        self.d_profile = QtWidgets.QComboBox()
        self.d_profile.currentIndexChanged.connect(self._detail_changed)
        form.addRow("Spec profile", self.d_profile)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Export"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("File name (empty = automatic)"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("Resolution"), 0, 2)
        self.map_widgets = {}
        for r, k in enumerate(MAP_KEYS, start=1):
            cb = QtWidgets.QCheckBox(MAP_LABELS[k])
            fe = QtWidgets.QLineEdit()
            rc = QtWidgets.QComboBox()
            rc.addItems(RES_SIZES)
            cb.toggled.connect(self._detail_changed)
            fe.textEdited.connect(self._detail_changed)
            rc.currentIndexChanged.connect(self._detail_changed)
            grid.addWidget(cb, r, 0)
            grid.addWidget(fe, r, 1)
            grid.addWidget(rc, r, 2)
            self.map_widgets[k] = (cb, fe, rc)
        grid.setColumnStretch(1, 1)
        form.addRow(grid)
        self.detail.setEnabled(False)
        lay.addWidget(self.detail)
        # ---- output
        grp = QtWidgets.QGroupBox("Output")
        form = QtWidgets.QFormLayout(grp)
        row = QtWidgets.QHBoxLayout()
        self.out_dir = QtWidgets.QLineEdit(self.settings.value("out_dir", ""))
        b = QtWidgets.QPushButton("...")
        b.setFixedWidth(30)
        b.clicked.connect(self._browse)
        row.addWidget(self.out_dir)
        row.addWidget(b)
        form.addRow("Folder", row)
        self.filetype = self._combo([("DDS (NVIDIA Texture Tools)", "dds"), ("PNG", "png"),
                                     ("TGA", "tga")], "filetype", "dds")
        self.filetype.currentIndexChanged.connect(self._update_dds_enabled)
        form.addRow("File type", self.filetype)
        self.fmt_d = self._combo(D_FORMATS, "fmt_d", "auto")
        form.addRow("_d format", self.fmt_d)
        self.fmt_n = self._combo(N_FORMATS, "fmt_n", "bc3")
        form.addRow("_n format", self.fmt_n)
        self.fmt_s = self._combo(S_FORMATS, "fmt_s", "bc1")
        form.addRow("_s format", self.fmt_s)
        self.mips = QtWidgets.QCheckBox("Generate mipmaps")
        self.mips.setChecked(self.settings.value("mips", "true") == "true")
        form.addRow(self.mips)
        self.quality = self._combo([(q.capitalize(), q) for q in QUALITIES], "quality",
                                   "production")
        form.addRow("DDS quality", self.quality)
        row = QtWidgets.QHBoxLayout()
        self.nvtt = QtWidgets.QLineEdit(self.settings.value("nvtt", DEFAULT_NVTT))
        b = QtWidgets.QPushButton("...")
        b.setFixedWidth(30)
        b.clicked.connect(self._browse_nvtt)
        row.addWidget(self.nvtt)
        row.addWidget(b)
        form.addRow("nvtt_export.exe", row)
        self.extra = QtWidgets.QLineEdit(self.settings.value("extra", ""))
        self.extra.setToolTip("Extra arguments passed to nvtt_export.exe")
        form.addRow("Extra NVTT args", self.extra)
        self._dds_widgets = [self.fmt_d, self.fmt_n, self.fmt_s, self.mips, self.quality,
                             self.nvtt, self.extra]

        self.normal = self._combo([("DirectX (Y-)", "DirectX"), ("OpenGL (Y+)", "OpenGL")],
                                  "normal", "DirectX")
        form.addRow("Normal format", self.normal)
        self.darken = QtWidgets.QDoubleSpinBox()
        self.darken.setRange(0.0, 1.0)
        self.darken.setSingleStep(0.05)
        self.darken.setValue(float(self.settings.value("darken", 0.9)))
        self.darken.setToolTip("Darkens metal in _d (GTA chrome uses a near-black diffuse). "
                               "0 = off, 1 = metal becomes black")
        form.addRow("_d metal darkening", self.darken)
        lay.addWidget(grp)
        self._update_dds_enabled()

        # ---- profile editor
        grp = QtWidgets.QGroupBox("Spec profiles")
        gl = QtWidgets.QVBoxLayout(grp)
        self.profile_combo = QtWidgets.QComboBox()
        self.profile_combo.currentIndexChanged.connect(self._load_profile_into_form)
        gl.addWidget(self.profile_combo)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("New", self._profile_new), ("Duplicate", self._profile_dup),
                         ("Rename", self._profile_rename), ("Delete", self._profile_delete),
                         ("Reset defaults", self._profile_reset)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        gl.addLayout(row)
        form = QtWidgets.QFormLayout()
        self.falloff_mode = QtWidgets.QComboBox()
        self.falloff_mode.addItem("Linear (G = 1 - roughness)", "linear")
        self.falloff_mode.addItem("Phong exponent / specFalloffMult", "phong")
        self.falloff_mode.currentIndexChanged.connect(self._form_changed)
        form.addRow("Falloff mapping", self.falloff_mode)
        self.fields = {}
        for key, label, lo, hi, step, tip in PROFILE_FIELDS:
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setDecimals(3)
            sb.setToolTip(tip)
            sb.valueChanged.connect(self._form_changed)
            form.addRow(label, sb)
            self.fields[key] = sb
        gl.addLayout(form)
        self.preview = QtWidgets.QLabel()
        self.preview.setToolTip("_s pixel this profile writes for a few typical surfaces")
        gl.addWidget(self.preview)
        lay.addWidget(grp)
        self._fill_profile_combo()

        self.export_btn = QtWidgets.QPushButton("Export GTA textures")
        self.export_btn.setMinimumHeight(32)
        self.export_btn.clicked.connect(self.export)
        lay.addWidget(self.export_btn)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        lay.addWidget(self.log)
        lay.addStretch(1)

    def _update_dds_enabled(self, *_):
        dds = self.filetype.currentData() == "dds"
        for w in self._dds_widgets:
            w.setEnabled(dds)

    # --------------------------------------------------------- profiles ---
    def _profile_names(self):
        return [p["name"] for p in self.profiles]

    def _current_profile(self):
        i = self.profile_combo.currentIndex()
        return self.profiles[i] if 0 <= i < len(self.profiles) else None

    def _fill_profile_combo(self, select=None):
        self._loading = True
        self.profile_combo.clear()
        self.profile_combo.addItems(self._profile_names())
        if select in self._profile_names():
            self.profile_combo.setCurrentText(select)
        self._loading = False
        self._load_profile_into_form()
        self._refresh_table_combos()

    def _load_profile_into_form(self, *_):
        p = self._current_profile()
        if p is None:
            return
        self._loading = True
        self.falloff_mode.setCurrentIndex(1 if p["falloff_mode"] == "phong" else 0)
        for key, sb in self.fields.items():
            sb.setValue(float(p[key]))
        self.fields["falloff_mult"].setEnabled(p["falloff_mode"] == "phong")
        self._loading = False
        self._update_preview()

    def _form_changed(self, *_):
        if self._loading:
            return
        p = self._current_profile()
        if p is None:
            return
        p["falloff_mode"] = self.falloff_mode.currentData()
        for key, sb in self.fields.items():
            p[key] = sb.value()
        self.fields["falloff_mult"].setEnabled(p["falloff_mode"] == "phong")
        save_profiles(self.profiles)
        self._update_preview()

    def _update_preview(self):
        p = self._current_profile()
        if p is None or not hasattr(self, "preview"):
            return
        rows = []
        for label, r, m in (("Smooth metal / chrome", 0.0, 1.0),
                            ("Smooth non-metal", 0.0, 0.0),
                            ("Rough non-metal", 1.0, 0.0)):
            c = (_to8(spec_intensity(r, m, p)), _to8(roughness_to_falloff(r, p)),
                 _to8(spec_fresnel(m, p)))
            rows.append(
                f'<tr><td><span style="background:rgb{c};">&nbsp;&nbsp;&nbsp;&nbsp;</span></td>'
                f'<td>{label}</td><td>&nbsp;{c[0]}, {c[1]}, {c[2]}</td></tr>')
        self.preview.setText("<table>" + "".join(rows) + "</table>")

    def _ask_name(self, title, default):
        name, ok = QtWidgets.QInputDialog.getText(self, title, "Profile name:", text=default)
        name = name.strip()
        if not ok or not name:
            return None
        if name in self._profile_names():
            QtWidgets.QMessageBox.warning(self, PLUGIN_TITLE, "A profile with that name exists.")
            return None
        return name

    def _profile_new(self):
        name = self._ask_name("New profile", "My shader")
        if name:
            p = dict(DEFAULT_PROFILES[0])
            p["name"] = name
            self.profiles.append(p)
            save_profiles(self.profiles)
            self._fill_profile_combo(name)

    def _profile_dup(self):
        cur = self._current_profile()
        name = cur and self._ask_name("Duplicate profile", cur["name"] + " copy")
        if name:
            p = dict(cur)
            p["name"] = name
            self.profiles.append(p)
            save_profiles(self.profiles)
            self._fill_profile_combo(name)

    def _profile_rename(self):
        cur = self._current_profile()
        name = cur and self._ask_name("Rename profile", cur["name"])
        if name:
            old = cur["name"]
            cur["name"] = name
            for info in self.rows.values():
                if info["profile"] == old:
                    info["profile"] = name
            save_profiles(self.profiles)
            self._fill_profile_combo(name)

    def _profile_delete(self):
        if len(self.profiles) <= 1:
            return
        cur = self._current_profile()
        if QtWidgets.QMessageBox.question(self, PLUGIN_TITLE, f"Delete '{cur['name']}'?") \
                != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.profiles.remove(cur)
        save_profiles(self.profiles)
        self._fill_profile_combo()

    def _profile_reset(self):
        if QtWidgets.QMessageBox.question(
                self, PLUGIN_TITLE, "Restore the built-in profiles? Your own profiles are kept.") \
                != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        mine = {p["name"]: p for p in self.profiles}
        for d in DEFAULT_PROFILES:
            mine[d["name"]] = dict(d)
        self.profiles = list(mine.values())
        save_profiles(self.profiles)
        self._fill_profile_combo()

    # ---------------------------------------------------- texture sets ---
    def _read_assignments(self):
        try:
            data = sp_project.Metadata(METADATA_CONTEXT).get("assignments")
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _normalize(self, saved, ts_name, default_res):
        """Build a full info dict from saved data (also reads the older format)."""
        names = self._profile_names()
        info = {"ts": ts_name, "export": bool(saved.get("export", True)),
                "name": saved.get("name") or ts_name,
                "profile": saved.get("profile") if saved.get("profile") in names else names[0],
                "maps": {k: {"on": True, "file": "", "res": default_res} for k in MAP_KEYS}}
        maps = saved.get("maps") if isinstance(saved.get("maps"), dict) else {}
        for k in MAP_KEYS:
            m = info["maps"][k]
            if isinstance(maps.get(k), dict):
                for key in ("on", "file", "res"):
                    if key in maps[k]:
                        m[key] = maps[k][key]
            elif "res_" + k in saved:            # v2: resolution combo with "Off"
                v = saved["res_" + k]
                m["on"] = v != "Off"
                if v in RES_SIZES:
                    m["res"] = v
            if m["res"] not in RES_SIZES:
                m["res"] = default_res
        return info

    @staticmethod
    def file_name(info, k):
        f = info["maps"][k]["file"].strip()
        if f:
            base, ext = os.path.splitext(f)
            return base if ext.lower() in (".dds", ".png", ".tga") else f
        return f"{info['name'].strip() or info['ts']}_{k}"

    @staticmethod
    def _summary(info):
        on = [f"_{k} {m['res']}" for k, m in info["maps"].items() if m["on"]]
        return ", ".join(on) if on else "no maps"

    def _save_assignments(self, *_):
        if self._loading or not self.rows or not sp_project.is_open():
            return
        data = {ts: {k: v for k, v in info.items() if k != "ts"}
                for ts, info in self.rows.items()}
        try:
            sp_project.Metadata(METADATA_CONTEXT).set("assignments", data)
        except Exception as e:
            sp_log.warning(f"{PLUGIN_TITLE}: could not store assignments: {e}")

    def refresh_texture_sets(self):
        self._loading = True
        prev = self.cur_ts
        self.table.setRowCount(0)
        self.rows, self.cur_ts = {}, None
        if sp_project.is_open():
            saved = self._read_assignments()
            for ts in sp_ts.all_texture_sets():
                ts_name = ts.name()
                try:
                    res = ts.get_resolution()
                    default_res = str(max(res.width, res.height))
                except Exception:
                    default_res = "2048"
                if default_res not in RES_SIZES:
                    default_res = "2048"
                info = self._normalize(saved.get(ts_name, {}), ts_name, default_res)
                self.rows[ts_name] = info
                r = self.table.rowCount()
                self.table.insertRow(r)
                item = QtWidgets.QTableWidgetItem(ts_name)
                item.setFlags((item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                              | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.CheckState.Checked if info["export"]
                                   else QtCore.Qt.CheckState.Unchecked)
                self.table.setItem(r, COL_TS, item)
                self.table.setItem(r, COL_NAME, QtWidgets.QTableWidgetItem(info["name"]))
                summ = QtWidgets.QTableWidgetItem(self._summary(info))
                summ.setFlags(summ.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, COL_MAPS, summ)
        self.table.resizeColumnsToContents()
        self._loading = False
        names = list(self.rows)
        if names:
            self.table.selectRow(names.index(prev) if prev in names else 0)
        else:
            self._load_detail()
        self._save_assignments()

    def _row_of(self, ts_name):
        for r in range(self.table.rowCount()):
            if self.table.item(r, COL_TS).text() == ts_name:
                return r
        return -1

    def _update_row(self, ts_name):
        r = self._row_of(ts_name)
        if r < 0:
            return
        was = self._loading
        self._loading = True
        info = self.rows[ts_name]
        self.table.item(r, COL_NAME).setText(info["name"])
        self.table.item(r, COL_MAPS).setText(self._summary(info))
        self._loading = was

    def _table_item_changed(self, item):
        if self._loading:
            return
        ts_name = self.table.item(item.row(), COL_TS).text()
        info = self.rows.get(ts_name)
        if info is None:
            return
        if item.column() == COL_TS:
            info["export"] = item.checkState() == QtCore.Qt.CheckState.Checked
        elif item.column() == COL_NAME:
            info["name"] = item.text().strip() or ts_name
            if ts_name == self.cur_ts:
                self._load_detail()
        self._save_assignments()

    def _selection_changed(self):
        rows = self.table.selectionModel().selectedRows()
        self.cur_ts = self.table.item(rows[0].row(), COL_TS).text() if rows else None
        self._load_detail()

    def _load_detail(self):
        info = self.rows.get(self.cur_ts)
        self.detail.setEnabled(info is not None)
        self.detail.setTitle(f"Selected: {self.cur_ts}" if info else "Selected texture set")
        if info is None:
            return
        was = self._loading
        self._loading = True
        self.d_name.setText(info["name"])
        self.d_profile.setCurrentText(info["profile"])
        for k, (cb, fe, rc) in self.map_widgets.items():
            m = info["maps"][k]
            cb.setChecked(m["on"])
            fe.setText(m["file"])
            rc.setCurrentText(m["res"])
        self._loading = was
        self._update_detail_state(info)

    def _update_detail_state(self, info):
        for k, (cb, fe, rc) in self.map_widgets.items():
            fe.setPlaceholderText(f"{info['name'].strip() or info['ts']}_{k}")
            fe.setEnabled(cb.isChecked())
            rc.setEnabled(cb.isChecked())

    def _detail_changed(self, *_):
        if self._loading:
            return
        info = self.rows.get(self.cur_ts)
        if info is None:
            return
        info["name"] = self.d_name.text().strip() or info["ts"]
        if self.d_profile.currentText():
            info["profile"] = self.d_profile.currentText()
        for k, (cb, fe, rc) in self.map_widgets.items():
            info["maps"][k] = {"on": cb.isChecked(), "file": fe.text(),
                               "res": rc.currentText()}
        self._update_detail_state(info)
        self._update_row(self.cur_ts)
        self._save_assignments()

    def _refresh_table_combos(self):
        """Refill the profile dropdown after profiles were added/renamed/deleted."""
        if not hasattr(self, "d_profile"):
            return
        names = self._profile_names()
        for info in self.rows.values():
            if info["profile"] not in names:
                info["profile"] = names[0]
        self._loading = True
        self.d_profile.clear()
        self.d_profile.addItems(names)
        self._loading = False
        self._load_detail()
        self._save_assignments()

    def _apply_res_all(self):
        res = self.res_all.currentText()
        for ts_name, info in self.rows.items():
            for m in info["maps"].values():
                m["res"] = res
            self._update_row(ts_name)
        self._load_detail()
        self._save_assignments()

    def _on_project_open(self, *_):
        self.refresh_texture_sets()

    def _on_project_close(self, *_):
        self._loading = True
        self.table.setRowCount(0)
        self.rows, self.cur_ts = {}, None
        self._loading = False
        self._load_detail()

    # ----------------------------------------------------------- export ---
    def _browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    def _browse_nvtt(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Find nvtt_export.exe", os.path.dirname(self.nvtt.text()),
            "nvtt_export (nvtt_export.exe);;Programs (*.exe)")
        if f:
            self.nvtt.setText(f)

    def _log(self, msg):
        self.log.appendPlainText(msg)
        sp_log.info(f"{PLUGIN_TITLE}: {msg}")
        QtWidgets.QApplication.processEvents()

    def export(self):
        if not sp_project.is_open():
            self._log("Open a project first.")
            return
        opts = {
            "out_dir": self.out_dir.text().strip(),
            "filetype": self.filetype.currentData(),
            "fmt_d": self.fmt_d.currentData(), "fmt_n": self.fmt_n.currentData(),
            "fmt_s": self.fmt_s.currentData(),
            "normal": self.normal.currentData(),
            "darken": self.darken.value(),
            "mips": self.mips.isChecked(),
            "quality": self.quality.currentData(),
            "nvtt": self.nvtt.text().strip().strip('"'),
            "extra": self.extra.text(),
        }
        if not opts["out_dir"]:
            self._log("Choose an output folder first.")
            return
        for key, val in opts.items():
            if key == "mips":
                val = "true" if val else "false"
            self.settings.setValue(key, val)

        by_name = {p["name"]: p for p in self.profiles}
        jobs, used = [], {}
        for ts_name, info in self.rows.items():
            if not info["export"] or not any(m["on"] for m in info["maps"].values()):
                continue
            files = {k: self.file_name(info, k) for k in MAP_KEYS}
            for k, m in info["maps"].items():
                if not m["on"]:
                    continue
                key = files[k].lower()
                if key in used:
                    self._log(f"Two maps would be written as '{files[k]}' "
                              f"({used[key]} and {ts_name}). Give them different names.")
                    return
                used[key] = ts_name
            jobs.append({
                "ts": ts_name, "name": info["name"], "files": files,
                "profile": dict(by_name.get(info["profile"], self.profiles[0])),
                "res": {k: (int(m["res"]) if m["on"] else None)
                        for k, m in info["maps"].items()}})
        if not jobs:
            self._log("Nothing ticked to export.")
            return

        self.log.clear()
        self.export_btn.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            written = export_gta(jobs, opts, log=self._log)
            self._log(f"Done - {len(written)} file(s) written to {opts['out_dir']}")
        except Exception as e:
            self._log(f"ERROR: {e}")
            sp_log.error(traceback.format_exc())
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.export_btn.setEnabled(True)

    def shutdown(self):
        for ev, cb in self._events:
            try:
                sp_event.DISPATCHER.disconnect(ev, cb)
            except Exception:
                pass


# ============================================================= plugin ===
_WIDGET = None


def start_plugin():
    global _WIDGET
    _WIDGET = GTASpecWidget()
    sp_ui.add_dock_widget(_WIDGET)


def close_plugin():
    global _WIDGET
    if _WIDGET is not None:
        _WIDGET.shutdown()
        sp_ui.delete_ui_element(_WIDGET)
        _WIDGET = None


if __name__ == "__main__":
    start_plugin()
