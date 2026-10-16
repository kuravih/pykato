"""Dataclass wrapper for apre reveal .rvl measurement files."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

ABERRATIONS = ("tilt", "power", "astigmatism", "coma", "spherical")
ANGLED = ("tilt", "astigmatism", "coma")

UNIT_KEYS = ("amount", "angle", "current", "intensity", "length", "mass", "solidangle", "temperature", "time")


def _read(g: h5py.Group, key: str, default=None):
    """Read a dataset as a plain Python/numpy value; bytes -> str, Empty -> default."""
    if key not in g:
        return default
    v = g[key][()]
    if isinstance(v, h5py.Empty):
        return default
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.generic):
        return v.item()
    return v


def _read_dict(g: h5py.Group, key: str) -> dict:
    if key not in g:
        return {}
    return {k: _read(g[key], k) for k in g[key] if isinstance(g[key][k], h5py.Dataset)}


@dataclass
class Axis:
    origin: float
    pixel_origin: float
    pixel_scale: float
    data_units: dict[str, str] = field(default_factory=dict)
    display_units: dict[str, str] = field(default_factory=dict)
    unit_dimensions: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_group(cls, g: h5py.Group) -> Axis:
        pt = g["pixeltransform"]
        return cls(origin=_read(g, "origin", 0.0), pixel_origin=_read(pt, "origin", 0.0), pixel_scale=_read(pt, "scale", 1.0), data_units=_read_dict(g, "dataunits"), display_units=_read_dict(g, "displayunits"), unit_dimensions=_read_dict(g, "unitdimensions"))

    def coords(self, n: int, pixel_size: float = 1.0) -> np.ndarray:
        """Physical coordinate of each of n pixels along this axis."""
        return (np.arange(n) - self.pixel_origin) * self.pixel_scale * pixel_size


@dataclass
class Result:
    data: np.ndarray  # float32 (n, H, W)
    mask: np.ndarray  # uint8 (H, W)
    mask_tree: bytes
    valuetree: str
    mask_xml: str
    filters_xml: str
    features_xml: str
    fiducials_xml: str
    exportformat: int
    x: Axis
    y: Axis
    z: Axis

    @classmethod
    def from_group(cls, g: h5py.Group) -> Result:
        mt = _read(g, "mask_tree")
        return cls(data=_read(g, "inputdata"), mask=_read(g, "inheritedmask"), mask_tree=bytes(mt) if mt is not None else b"", valuetree=_read(g, "valuetree", ""), mask_xml=_read(g, "mask_string", ""), filters_xml=_read(g, "filter_string", ""), features_xml=_read(g, "feature_string", ""), fiducials_xml=_read(g, "fiducials_string", ""), exportformat=_read(g, "exportformat", 0), x=Axis.from_group(g["xaxis"]), y=Axis.from_group(g["yaxis"]), z=Axis.from_group(g["zaxis"]))

    @property
    def phase(self) -> np.ndarray:
        return self.data[0]

    def valid(self, maskbits: int | None = None) -> np.ndarray:
        """True where the pixel is NOT masked out (mask bits mark invalid pixels)."""
        if maskbits is None:
            return self.mask == 0
        return (self.mask & maskbits) == 0

    def extent(self, pixel_size: float = 1.0) -> tuple[float, float, float, float]:
        """(left, right, bottom, top) for matplotlib imshow with origin='lower'."""
        h, w = self.mask.shape
        xc = self.x.coords(w, pixel_size)
        yc = self.y.coords(h, pixel_size)
        return xc[0], xc[-1], yc[0], yc[-1]


@dataclass
class Analysis:
    name: str
    class_name: str
    description: str
    dataid: int
    enabled: bool
    scriptid: str
    maskbits: int
    prototype: bool
    inputresultname: str
    showonlyaveraged: bool
    usevalidators: bool
    mainresult: Result
    resultarray: Result

    @classmethod
    def from_group(cls, g: h5py.Group) -> Analysis:
        return cls(name=_read(g, "name", ""), class_name=_read(g, "class", ""), description=_read(g, "description", ""), dataid=_read(g, "dataid", 0), enabled=bool(_read(g, "enabled", 0)), scriptid=_read(g, "scriptid", ""), maskbits=_read(g, "maskbits", 0), prototype=bool(_read(g, "prototype", 0)), inputresultname=_read(g, "inputresultname", ""), showonlyaveraged=bool(_read(g, "showonlyaveraged", 0)), usevalidators=bool(_read(g, "usevalidators", 0)), mainresult=Result.from_group(g["mainresult"]), resultarray=Result.from_group(g["resultarray"]))


# ---------------------------------------------------------------- inittree


def _f(e: ET.Element, tag: str, default=None):
    t = e.findtext(tag)
    return default if t is None else float(t)


@dataclass
class UnitizedValue:
    value: float
    offset: float = 0.0
    tolerance_type: int = 0
    low_tolerance: float = 0.0
    high_tolerance: float = 0.0
    unit_dimension: dict[str, float] = field(default_factory=dict)
    modified: str | None = None

    @classmethod
    def parse(cls, xml: str) -> UnitizedValue:
        e = ET.fromstring(xml)
        ud = e.find("unitdimension")
        return cls(value=_f(e, "value"), offset=_f(e, "offset", 0.0), tolerance_type=int(_f(e, "tolerancetype", 0)), low_tolerance=_f(e, "lowtolerance", 0.0), high_tolerance=_f(e, "hightolerance", 0.0), unit_dimension={c.tag: float(c.text) for c in ud} if ud is not None else {}, modified=e.findtext("modified"))


def _node(tree: ET.Element, guid: str) -> dict[str, str | None]:
    """Raw <value> text of each direct child key of a node in a valuetree."""
    n = tree.find(f"key[@name='{guid}']")
    if n is None:
        return {}
    return {k.get("name"): k.findtext("value") for k in n.findall("key")}


def _uv(raw: dict, key: str) -> UnitizedValue | None:
    t = raw.get(key)
    return UnitizedValue.parse(t) if t and t.lstrip().startswith("<") else None


def _num(raw: dict, key: str) -> float | None:
    u = _uv(raw, key)
    return None if u is None else u.value


@dataclass
class SurfaceStats:
    """BasicInterfStatsAnalysis output (vertical values in the file's vertical units)."""

    pv: float
    rms: float
    mean: float
    absmean: float
    peak: float
    valley: float
    num_valid_points: int
    data_width: int
    data_height: int
    x_range: tuple[int, int]
    y_range: tuple[int, int]
    raw: dict[str, UnitizedValue] = field(default_factory=dict, repr=False)

    @classmethod
    def from_raw(cls, raw: dict) -> SurfaceStats:
        u = {k: _uv(raw, k) for k in raw}
        v = {k: x.value for k, x in u.items() if x is not None}
        return cls(pv=v["pv"], rms=v["rms"], mean=v["mean"], absmean=v["absmean"], peak=v["peak"], valley=v["valley"], num_valid_points=int(v["numvalidpoints"]), data_width=int(v["datawidth"]), data_height=int(v["dataheight"]), x_range=(int(v["xdatastart"]), int(v["xdataend"])), y_range=(int(v["ydatastart"]), int(v["ydataend"])), raw={k: x for k, x in u.items() if x is not None})


@dataclass
class Aberrations:
    tilt: float
    tilt_angle: float
    power: float
    astigmatism: float
    astigmatism_angle: float
    coma: float
    coma_angle: float
    spherical: float

    @classmethod
    def from_raw(cls, raw: dict, prefix: str) -> Aberrations:
        kw = {a: _num(raw, prefix + a) for a in ABERRATIONS}
        kw |= {f"{a}_angle": _num(raw, prefix + a + "angle") for a in ANGLED}
        return cls(**kw)


@dataclass
class ZernikeFit:
    num_terms: int
    removed_terms: list[bool]
    data_diameter: float  # pixels
    unit_circle_center: tuple[float, float]  # pixels
    power_deviation: float
    input_power_deviation: float
    zernike: Aberrations  # after term removal
    input_zernike: Aberrations  # before term removal
    seidel: Aberrations

    @classmethod
    def from_raw(cls, raw: dict) -> ZernikeFit:
        n = int(raw.get("numterms") or 0)
        rt = [t.strip() == "true" for t in (raw.get("removeterms") or "").split(",") if t]
        return cls(num_terms=n, removed_terms=rt[:n], data_diameter=_num(raw, "datadiameter"), unit_circle_center=(_num(raw, "unitcirclecenterx"), _num(raw, "unitcirclecentery")), power_deviation=_num(raw, "powerdeviation"), input_power_deviation=_num(raw, "inputpowerdeviation"), zernike=Aberrations.from_raw(raw, "zernike"), input_zernike=Aberrations.from_raw(raw, "inputzernike"), seidel=Aberrations.from_raw(raw, "seidel"))


@dataclass
class PVqSettings:
    target_q: float
    bin_count: int

    @classmethod
    def from_raw(cls, raw: dict) -> PVqSettings:
        return cls(target_q=float(raw["targetq"]), bin_count=int(raw["bincount"]))


def _graph_nodes(graph_xml: str) -> dict[str, str]:
    """class name -> node GUID, from the analysis graph."""
    if not graph_xml:
        return {}
    return {a.get("class"): a.get("id") for a in ET.fromstring(graph_xml).iter("analysis")}


@dataclass
class MeasurementFile:
    path: Path
    commondata: str
    graph: str
    inittree: str
    main: Result
    root: Analysis
    analyses: dict[str, Analysis | None]
    nodes: dict[str, str] = field(default_factory=dict)  # class name -> GUID
    stats: SurfaceStats | None = None
    zernike: ZernikeFit | None = None
    pvq: PVqSettings | None = None

    @classmethod
    def load(cls, path: str | Path) -> MeasurementFile:
        path = Path(path)
        with h5py.File(path, "r") as f:
            analyses = {k: Analysis.from_group(f[k]) if "class" in f[k] else None for k in f if k.startswith("{")}
            graph = _read(f, "graph", "")
            inittree = _read(f, "inittree", "")
            nodes = _graph_nodes(graph)
            tree = ET.fromstring(inittree) if inittree else None

            def parsed(cls_name, parser):
                guid = nodes.get(cls_name)
                raw = _node(tree, guid) if tree is not None and guid else {}
                return parser(raw) if raw else None

            return cls(path=path, commondata=_read(f, "commondata", ""), graph=graph, inittree=inittree, nodes=nodes, stats=parsed("aox::BasicInterfStatsAnalysis", SurfaceStats.from_raw), zernike=parsed("aox::ZernikeFitAnalysis", ZernikeFit.from_raw), pvq=parsed("aox::PVqAnalysis", PVqSettings.from_raw), main=Result.from_group(f["mainresult"]), root=Analysis.from_group(f["rootanalysis"]), analyses=analyses)

    def config(self, *keys: str) -> str | None:
        """Raw string value at a key path in commondata, e.g. config('measurementconfig', 'pixelsize')."""
        node = ET.fromstring(self.commondata)
        for k in keys:
            node = node.find(f"key[@name='{k}']")
            if node is None:
                return None
        v = node.find("value")
        return v.text if v is not None else None

    @property
    def wavelength(self) -> float:
        return float(self.config("measurementconfig", "wavelength"))

    @property
    def pixel_size(self) -> float:
        return float(self.config("measurementconfig", "pixelsize"))
