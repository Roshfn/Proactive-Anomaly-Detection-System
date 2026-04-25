# modules package — lazy imports so unit tests work without torch/ultralytics
__all__ = ["Tracklet", "cascade_match", "SurveillanceModule", "ArsonModule"]

def __getattr__(name):
    if name == "Tracklet":
        from modules.tracklet import Tracklet
        return Tracklet
    if name == "cascade_match":
        from modules.matching import cascade_match
        return cascade_match
    if name == "SurveillanceModule":
        from modules.surveillance_module import SurveillanceModule
        return SurveillanceModule
    if name == "ArsonModule":
        from modules.arson_module import ArsonModule
        return ArsonModule
    raise AttributeError(f"module 'modules' has no attribute {name!r}")
