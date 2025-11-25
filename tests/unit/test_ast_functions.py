"""
Unit tests for GLSL built-in function transformations (Session 3).

Tests all GLSL built-in functions (excluding matrix operations which are Session 4-5).
Verifies that function calls are correctly transformed to GLSL_ prefix using AST-based approach.

Test Structure:
- Trigonometric functions (10 tests)
- Exponential/Power functions (8 tests)
- Common/Math functions (14 tests)
- Geometric functions (8 tests)
- Hyperbolic functions (3 tests)

Total: 43+ unit tests
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import (
    TypeChecker,
    create_builtin_symbol_table,
)
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.transformer.code_emitter import CodeEmitter


@pytest.fixture
def transformer():
    """Create transformer with parser and type checker."""
    parser = GLSLParser()
    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)
    return ASTTransformer(type_checker), parser, CodeEmitter()


# ========================================================================
# Trigonometric Functions (10 tests)
# ========================================================================

def test_sin_function(transformer):
    """Test sin() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = sin(1.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_sin(1.0f)" in opencl


def test_cos_function(transformer):
    """Test cos() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = cos(1.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_cos(1.0f)" in opencl


def test_tan_function(transformer):
    """Test tan() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = tan(0.5); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_tan(0.5f)" in opencl


def test_asin_function(transformer):
    """Test asin() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = asin(0.5); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_asin(0.5f)" in opencl


def test_acos_function(transformer):
    """Test acos() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = acos(0.5); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_acos(0.5f)" in opencl


def test_atan_single_arg(transformer):
    """Test atan(y_over_x) function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = atan(1.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_atan(1.0f)" in opencl


def test_atan_two_args(transformer):
    """Test atan(y, x) function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = atan(1.0, 2.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_atan(1.0f, 2.0f)" in opencl


def test_radians_function(transformer):
    """Test radians() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = radians(180.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_radians(180.0f)" in opencl


def test_degrees_function(transformer):
    """Test degrees() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = degrees(3.14159); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_degrees(3.14159f)" in opencl


def test_trig_with_vectors(transformer):
    """Test trig functions with vector arguments."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { vec3 v = vec3(1.0); vec3 x = sin(v); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_sin" in opencl
    assert "float3" in opencl


# ========================================================================
# Exponential/Power Functions (8 tests)
# ========================================================================

def test_pow_function(transformer):
    """Test pow() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = pow(2.0, 3.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_pow(2.0f, 3.0f)" in opencl


def test_exp_function(transformer):
    """Test exp() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = exp(2.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_exp(2.0f)" in opencl


def test_log_function(transformer):
    """Test log() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = log(10.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_log(10.0f)" in opencl


def test_exp2_function(transformer):
    """Test exp2() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = exp2(3.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_exp2(3.0f)" in opencl


def test_log2_function(transformer):
    """Test log2() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = log2(8.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_log2(8.0f)" in opencl


def test_sqrt_function(transformer):
    """Test sqrt() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = sqrt(4.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_sqrt(4.0f)" in opencl


def test_inversesqrt_function(transformer):
    """Test inversesqrt() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = inversesqrt(4.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_inversesqrt(4.0f)" in opencl


def test_pow_with_vectors(transformer):
    """Test pow() with vector arguments."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { vec3 v = pow(vec3(2.0), vec3(3.0)); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_pow" in opencl
    assert "float3" in opencl


# ========================================================================
# Common/Math Functions (14 tests)
# ========================================================================

def test_abs_function(transformer):
    """Test abs() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = abs(-5.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_abs" in opencl


def test_sign_function(transformer):
    """Test sign() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = sign(-3.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_sign" in opencl


def test_floor_function(transformer):
    """Test floor() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = floor(3.7); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_floor(3.7f)" in opencl


def test_ceil_function(transformer):
    """Test ceil() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = ceil(3.2); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_ceil(3.2f)" in opencl


def test_fract_function(transformer):
    """Test fract() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = fract(3.7); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_fract(3.7f)" in opencl


def test_mod_function(transformer):
    """Test mod() function transformation (GLSL semantics)."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = mod(5.5, 2.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_mod(5.5f, 2.0f)" in opencl


def test_min_function(transformer):
    """Test min() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = min(3.0, 5.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_min(3.0f, 5.0f)" in opencl


def test_max_function(transformer):
    """Test max() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = max(3.0, 5.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_max(3.0f, 5.0f)" in opencl


def test_clamp_function(transformer):
    """Test clamp() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = clamp(5.0, 0.0, 1.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_clamp(5.0f, 0.0f, 1.0f)" in opencl


def test_mix_function(transformer):
    """Test mix() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = mix(0.0, 1.0, 0.5); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_mix(0.0f, 1.0f, 0.5f)" in opencl


def test_step_function(transformer):
    """Test step() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = step(0.5, 0.7); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_step(0.5f, 0.7f)" in opencl


def test_smoothstep_function(transformer):
    """Test smoothstep() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = smoothstep(0.0, 1.0, 0.5); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_smoothstep(0.0f, 1.0f, 0.5f)" in opencl


def test_trunc_function(transformer):
    """Test trunc() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = trunc(3.7); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_trunc(3.7f)" in opencl


def test_min_with_vectors(transformer):
    """Test min() with vector arguments."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { vec3 x = min(vec3(1.0), vec3(0.5)); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_min" in opencl
    assert "float3" in opencl


# ========================================================================
# Geometric Functions (8 tests)
# ========================================================================

def test_length_function(transformer):
    """Test length() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = length(vec3(1.0, 2.0, 3.0)); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_length" in opencl


def test_distance_function(transformer):
    """Test distance() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = distance(vec2(0.0), vec2(1.0)); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_distance" in opencl


def test_dot_function(transformer):
    """Test dot() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = dot(vec3(1.0), vec3(2.0)); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_dot" in opencl


def test_cross_function(transformer):
    """Test cross() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { vec3 x = cross(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_cross" in opencl


def test_normalize_function(transformer):
    """Test normalize() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { vec3 x = normalize(vec3(1.0, 2.0, 3.0)); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_normalize" in opencl


def test_reflect_function(transformer):
    """Test reflect() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { vec3 x = reflect(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_reflect" in opencl


def test_refract_function(transformer):
    """Test refract() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { vec3 x = refract(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), 0.9); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_refract" in opencl


def test_faceforward_function(transformer):
    """Test faceforward() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { vec3 x = faceforward(vec3(1.0), vec3(0.5), vec3(0.3)); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_faceforward" in opencl


# ========================================================================
# Hyperbolic Functions (3 tests)
# ========================================================================

def test_sinh_function(transformer):
    """Test sinh() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = sinh(1.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_sinh(1.0f)" in opencl


def test_cosh_function(transformer):
    """Test cosh() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = cosh(1.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_cosh(1.0f)" in opencl


def test_tanh_function(transformer):
    """Test tanh() function transformation."""
    ast_transformer, parser, emitter = transformer
    source = "void main() { float x = tanh(1.0); }"

    ast = parser.parse(source)
    transformed_ast = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed_ast)

    assert "GLSL_tanh(1.0f)" in opencl
