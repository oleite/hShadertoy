"""
Test Suite for hShadertoy GLSL to OpenCL Transpiler

Test organization:
    - unit/: Module-level tests (2000+ target)
    - integration/: Cross-module tests (300+ target)
    - fixtures/: Test data and reference shaders
    - helpers/: Test utilities (kernel execution, image comparison)

Test Strategy:
    - Test-Driven Development (TDD): Tests written BEFORE implementation
    - Coverage Target: >85% overall, >90% for critical modules
    - Success Rate Target: >70% of real Shadertoy shaders transpile correctly

Testing Tools:
    - pytest: Test framework
    - pytest-cov: Coverage reporting
    - PyOpenCL: OpenCL kernel compilation testing
    - numpy: Numerical validation
"""
