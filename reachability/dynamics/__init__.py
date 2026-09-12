import importlib

# Submodule holding each dynamics class, keyed by class name. Classes are
# imported lazily (via module __getattr__ below) instead of all up front.
_MODULE_FOR_CLASS = {
    "Dynamics":                   "base",
    "Dubins3D":                   "dubins3d",
    "NarrowPassage":              "narrow_passage",
    "PursuitEvasion":             "pursuer_evader",
    "QuadrotorGateTraversal":     "gate_traversal",
    "QuadrotorGateAvoidance":     "quadrotor_avoid",
    "QuadrotorCylinderAvoidance": "quadrotor_cylinder_avoid",
    "LessLinearND":               "leslinear_nd",
    "F1TenthBEV":                 "f1tenth_bev",
}

__all__ = list(_MODULE_FOR_CLASS)


def __getattr__(name: str):
    module_name = _MODULE_FOR_CLASS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache so repeated access skips __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_MODULE_FOR_CLASS))
