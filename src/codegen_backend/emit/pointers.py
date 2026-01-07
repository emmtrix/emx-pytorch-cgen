from __future__ import annotations


def emit_input_ptr(var_name: str, c_type: str, const: bool = True) -> str:
    const_prefix = "const " if const else ""
    ptr_name = f"{var_name}_ptr"
    return (
        f"{const_prefix}{c_type}* {ptr_name} = "
        f"({const_prefix}{c_type}*){var_name};"
    )


def emit_output_ptr(var_name: str, c_type: str) -> str:
    return emit_input_ptr(var_name, c_type, const=False)
