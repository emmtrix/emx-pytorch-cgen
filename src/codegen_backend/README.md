# Codegen Backend

The codegen backend is responsible for lowering supported PyTorch operators to
simple, analyzable C. It is the primary implementation area for adding or
extending operators.

## Key files

- `backend.py`: Operator registration, supported ops list, dtype coverage, and
  target mappings.
- `templates/`: Jinja templates for emitting C code. Prefer updating templates
  here over inline string assembly.

## Adding an operator

1. Register the operator in `backend.py` using `_binary_spec`, `_unary_spec`, or
   a custom `_OpSpec`.
2. Add coverage in `tests/test_codegen_ops.py`.
3. Ensure targets are mapped in `TARGET_REGISTRY` and wire in-place behavior if
   applicable.
4. Update dtype coverage as needed.
5. If the operator needs new shapes or emission logic, add or update a Jinja
   template under `templates/`.
