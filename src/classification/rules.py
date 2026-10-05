"""
General Rules for Interpretation used by the classification workflow.

These constants identify the GRI rules referenced by the classification
reasoning. The actual application of a rule must be supported by retrieved
tariff evidence.
"""

GRI_1 = "GRI 1"
GRI_2A = "GRI 2(a)"
GRI_2B = "GRI 2(b)"
GRI_3A = "GRI 3(a)"
GRI_3B = "GRI 3(b)"
GRI_3C = "GRI 3(c)"
GRI_4 = "GRI 4"
GRI_5A = "GRI 5(a)"
GRI_5B = "GRI 5(b)"
GRI_6 = "GRI 6"


SUPPORTED_GRI_RULES = (
    GRI_1,
    GRI_2A,
    GRI_2B,
    GRI_3A,
    GRI_3B,
    GRI_3C,
    GRI_4,
    GRI_5A,
    GRI_5B,
    GRI_6,
)