from __future__ import annotations
import pandera as pa
from pandera import Column, Check

# After aliasing, masters must expose these canonical columns
RtiMasterSchema = pa.DataFrameSchema(
    {
        "AC_MSN": Column(str, nullable=False, coerce=True),
        "gti_number": Column(str, nullable=False, coerce=True),
        "rti_number": Column(str, nullable=False, coerce=True),
        "title": Column(str, nullable=False, checks=Check.str_length(min_value=1), coerce=True),
        "workload": Column(int, nullable=True, coerce=True),
    },
    strict=False,
)

# After aliasing, chapters must expose these canonical columns
RtiChaptersSchema = pa.DataFrameSchema(
    {
        "AC_MSN": Column(str, nullable=False, coerce=True),
        "gti_number": Column(str, nullable=False, coerce=True),
        "rti_number": Column(str, nullable=False, coerce=True),
        "chapter_number": Column(int, nullable=False, checks=Check.ge(1), coerce=True),
        "chapter_title": Column(str, nullable=False, checks=Check.str_length(min_value=1), coerce=True),
        "attestation_type": Column(str, nullable=True, coerce=True),
    },
    strict=False,
)
