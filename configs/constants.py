from enum import StrEnum

PROJECT_NAME = "HJ_racing"

class ProblemType(StrEnum):
    BRT  = "BRT"
    BRS  = "BRS"
    BRAT = "BRAT"

class Role(StrEnum):
    MAX = "max"
    MIN = "min"

class DynamicsType(StrEnum):
    DUBINS_3D        = "Dubins3D"
    NARROW_PASSAGE   = "NarrowPassage"
    PURSUIT_EVASION  = "PursuitEvasion"
    GATE_TRAVERSAL   = "QuadrotorGateTraversal"
    GATE_AVOIDANCE   = "QuadrotorGateAvoidance"
    CYLINDER_AVOID   = "QuadrotorCylinderAvoidance"
    LESS_LINEAR_ND   = "LessLinearND"
    F1_TENTH_BEV     = "F1TenthBEV"

class VALUE_ARCH(StrEnum):
    MULTINET    = "valmultinet"

class POLICY_ARCH(StrEnum):
    VQ_MULTINET   = "vqpolmultinet"

