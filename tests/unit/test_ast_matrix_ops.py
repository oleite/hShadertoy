"""
Unit tests for matrix operation transformations (Session 4).

Tests binary expression transformation for matrix operations:
- Matrix * Vector
- Vector * Matrix
- Matrix * Matrix
- Compound assignments (*=)

All tests verify AST transformation correctness.
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import TypeChecker, create_builtin_symbol_table
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.transformer.code_emitter import CodeEmitter


@pytest.fixture
def transformer():
    """Create transformer with type checker."""
    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)
    return ASTTransformer(type_checker), CodeEmitter()


# ============================================================================
# mat2 Operations
# ============================================================================

def test_mat2_vector_multiply(transformer):
    """Test mat2 * vec2 multiplication."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        mat2 M = mat2(1.0);
        vec2 v = vec2(1.0, 0.0);
        vec2 result = M * v;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Transformed to typed GLSL_mul function
    assert 'GLSL_mul_mat2_vec2(M, v)' in opencl
    assert 'float2 result = GLSL_mul_mat2_vec2(M, v)' in opencl


def test_vec2_matrix_multiply(transformer):
    """Test vec2 * mat2 multiplication."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        vec2 v = vec2(1.0, 0.0);
        mat2 M = mat2(1.0);
        vec2 result = v * M;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Transformed to typed GLSL_mul function
    assert 'GLSL_mul_vec2_mat2(v, M)' in opencl
    assert 'float2 result = GLSL_mul_vec2_mat2(v, M)' in opencl


def test_mat2_mat2_multiply(transformer):
    """Test mat2 * mat2 multiplication."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        mat2 M1 = mat2(1.0);
        mat2 M2 = mat2(2.0);
        mat2 result = M1 * M2;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Transformed to typed GLSL_mul function
    assert 'GLSL_mul_mat2_mat2(M1, M2)' in opencl
    assert 'matrix2x2 result = GLSL_mul_mat2_mat2(M1, M2)' in opencl


def test_vec2_compound_assignment(transformer):
    """Test vec2 *= mat2 compound assignment."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        vec2 v = vec2(1.0, 0.0);
        mat2 M = mat2(1.0);
        v *= M;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Compound assignment transformed to regular assignment with GLSL_mul
    assert 'v = GLSL_mul_vec2_mat2(v, M)' in opencl


def test_mat2_chained_multiply(transformer):
    """Test chained mat2 operations: v * M1 * M2."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        vec2 v = vec2(1.0, 0.0);
        mat2 M1 = mat2(1.0);
        mat2 M2 = mat2(2.0);
        vec2 result = v * M1 * M2;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Chained operations transformed to nested GLSL_mul calls
    assert 'GLSL_mul_vec2_mat2(GLSL_mul_vec2_mat2(v, M1), M2)' in opencl


# ============================================================================
# mat3 Operations
# ============================================================================

def test_mat3_vector_multiply(transformer):
    """Test mat3 * vec3 multiplication."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        mat3 M = mat3(1.0);
        vec3 v = vec3(1.0, 0.0, 0.0);
        vec3 result = M * v;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Transformed to typed GLSL_mul function
    assert 'GLSL_mul_mat3_vec3(M, v)' in opencl
    assert 'float3 result = GLSL_mul_mat3_vec3(M, v)' in opencl


def test_vec3_matrix_multiply(transformer):
    """Test vec3 * mat3 multiplication."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        vec3 v = vec3(1.0, 0.0, 0.0);
        mat3 M = mat3(1.0);
        vec3 result = v * M;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Transformed to typed GLSL_mul function
    assert 'GLSL_mul_vec3_mat3(v, M)' in opencl
    assert 'float3 result = GLSL_mul_vec3_mat3(v, M)' in opencl


def test_mat3_mat3_multiply_declaration(transformer):
    """Test mat3 * mat3 multiplication with declaration."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        mat3 M1 = mat3(1.0);
        mat3 M2 = mat3(2.0);
        mat3 result = M1 * M2;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Transformed to typed GLSL_mul function
    assert 'GLSL_mul_mat3_mat3(M1, M2)' in opencl
    assert 'matrix3x3 result = GLSL_mul_mat3_mat3(M1, M2)' in opencl


def test_vec3_compound_assignment(transformer):
    """Test vec3 *= mat3 compound assignment."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        vec3 v = vec3(1.0, 0.0, 0.0);
        mat3 M = mat3(1.0);
        v *= M;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Compound assignment transformed to regular assignment with GLSL_mul
    assert 'v = GLSL_mul_vec3_mat3(v, M)' in opencl


def test_mat3_chained_multiply(transformer):
    """Test chained mat3 operations: v * M1 * M2."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        vec3 v = vec3(1.0, 0.0, 0.0);
        mat3 M1 = mat3(1.0);
        mat3 M2 = mat3(2.0);
        vec3 result = v * M1 * M2;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Chained operations transformed to nested GLSL_mul calls
    assert 'GLSL_mul_vec3_mat3(GLSL_mul_vec3_mat3(v, M1), M2)' in opencl


# ============================================================================
# mat4 Operations
# ============================================================================

def test_mat4_vector_multiply(transformer):
    """Test mat4 * vec4 multiplication."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        mat4 M = mat4(1.0);
        vec4 v = vec4(1.0, 0.0, 0.0, 0.0);
        vec4 result = M * v;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Transformed to typed GLSL_mul function
    assert 'GLSL_mul_mat4_vec4(M, v)' in opencl
    assert 'float4 result = GLSL_mul_mat4_vec4(M, v)' in opencl


def test_vec4_matrix_multiply(transformer):
    """Test vec4 * mat4 multiplication."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        vec4 v = vec4(1.0, 0.0, 0.0, 0.0);
        mat4 M = mat4(1.0);
        vec4 result = v * M;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Transformed to typed GLSL_mul function
    assert 'GLSL_mul_vec4_mat4(v, M)' in opencl
    assert 'float4 result = GLSL_mul_vec4_mat4(v, M)' in opencl


def test_mat4_mat4_multiply(transformer):
    """Test mat4 * mat4 multiplication."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        mat4 M1 = mat4(1.0);
        mat4 M2 = mat4(2.0);
        mat4 result = M1 * M2;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Transformed to typed GLSL_mul function
    assert 'GLSL_mul_mat4_mat4(M1, M2)' in opencl
    assert 'matrix4x4 result = GLSL_mul_mat4_mat4(M1, M2)' in opencl


def test_vec4_compound_assignment(transformer):
    """Test vec4 *= mat4 compound assignment."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        vec4 v = vec4(1.0, 0.0, 0.0, 0.0);
        mat4 M = mat4(1.0);
        v *= M;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Compound assignment transformed to regular assignment with GLSL_mul
    assert 'v = GLSL_mul_vec4_mat4(v, M)' in opencl


def test_mat4_chained_multiply(transformer):
    """Test chained mat4 operations: v * M1 * M2."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        vec4 v = vec4(1.0, 0.0, 0.0, 0.0);
        mat4 M1 = mat4(1.0);
        mat4 M2 = mat4(2.0);
        vec4 result = v * M1 * M2;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Chained operations transformed to nested GLSL_mul calls
    assert 'GLSL_mul_vec4_mat4(GLSL_mul_vec4_mat4(v, M1), M2)' in opencl


# ============================================================================
# Mixed Operations
# ============================================================================

def test_matrix_multiply_in_expression(transformer):
    """Test matrix multiplication within larger expression."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        mat2 M = mat2(1.0);
        vec2 v = vec2(1.0, 0.0);
        vec2 result = M * v + vec2(0.5);
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Transformed to typed GLSL_mul function
    assert 'GLSL_mul_mat2_vec2(M, v)' in opencl


def test_matrix_multiply_as_function_argument(transformer):
    """Test matrix multiplication as function argument."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        mat2 M = mat2(1.0);
        vec2 v = vec2(1.0, 0.0);
        float len = length(M * v);
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Transformed to typed GLSL_mul function
    assert 'GLSL_mul_mat2_vec2(M, v)' in opencl
    assert 'GLSL_length' in opencl


def test_multiple_matrix_operations(transformer):
    """Test multiple independent matrix operations."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        mat2 M = mat2(1.0);
        vec2 v1 = vec2(1.0, 0.0);
        vec2 v2 = vec2(0.0, 1.0);
        vec2 r1 = M * v1;
        vec2 r2 = v2 * M;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Transformed to typed GLSL_mul functions
    assert 'GLSL_mul_mat2_vec2(M, v1)' in opencl
    assert 'GLSL_mul_vec2_mat2(v2, M)' in opencl


def test_mat2_full_constructor_with_multiply(transformer):
    """Test mat2 full constructor followed by multiplication."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        mat2 M = mat2(1.0, 0.0, 0.0, 1.0);
        vec2 v = vec2(1.0, 0.0);
        vec2 result = M * v;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # mat2 full constructor uses GLSL_mat2() function
    assert 'GLSL_mat2(1.0f, 0.0f, 0.0f, 1.0f)' in opencl
    assert 'GLSL_mul_mat2_vec2(M, v)' in opencl


def test_mat3_full_constructor_with_multiply(transformer):
    """Test mat3 full constructor followed by multiplication."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        mat3 M = mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        vec3 v = vec3(1.0, 0.0, 0.0);
        vec3 result = M * v;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # mat3 full constructor uses GLSL_mat3() function, multiplication uses typed GLSL_mul
    assert 'GLSL_mul_mat3_vec3(M, v)' in opencl


def test_compound_with_parentheses(transformer):
    """Test compound assignment with parenthesized expression."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        vec2 v = vec2(1.0, 0.0);
        mat2 M1 = mat2(1.0);
        mat2 M2 = mat2(2.0);
        v *= (M1 * M2);
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Compound assignment with matrix multiply becomes regular multiply with typed function
    assert 'GLSL_mul_mat2_mat2(M1, M2)' in opencl


def test_sequential_compound_assignments(transformer):
    """Test multiple sequential compound assignments."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        vec2 v = vec2(1.0, 0.0);
        mat2 M1 = mat2(1.0);
        mat2 M2 = mat2(2.0);
        v *= M1;
        v *= M2;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Compound assignments transformed to regular assignments
    assert 'v = GLSL_mul_vec2_mat2(v, M1)' in opencl
    assert 'v = GLSL_mul_vec2_mat2(v, M2)' in opencl


def test_mat3_diagonal_with_multiply(transformer):
    """Test mat3 diagonal constructor followed by multiplication."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        mat3 M = mat3(2.0);
        vec3 v = vec3(1.0, 0.0, 0.0);
        vec3 result = M * v;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # mat3 diagonal uses GLSL_matrix3x3_diagonal, multiplication uses typed GLSL_mul
    assert 'GLSL_matrix3x3_diagonal' in opencl
    assert 'GLSL_mul_mat3_vec3(M, v)' in opencl


def test_matrix_in_return_statement(transformer):
    """Test matrix multiplication in return statement."""
    ast_transformer, emitter = transformer
    glsl = """
    vec2 transform(mat2 M, vec2 v) {
        return M * v;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    assert 'return GLSL_mul_mat2_vec2(M, v)' in opencl


def test_matrix_in_if_condition(transformer):
    """Test matrix multiplication result used in condition."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        mat2 M = mat2(1.0);
        vec2 v = vec2(1.0, 0.0);
        if (length(M * v) > 0.5) {
            v = vec2(0.0);
        }
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Transformed to typed GLSL_mul function
    assert 'GLSL_mul_mat2_vec2(M, v)' in opencl
    assert 'GLSL_length' in opencl


def test_matrix_array_access_with_multiply(transformer):
    """Test matrix array element access followed by multiplication."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        mat2 matrices[2];
        matrices[0] = mat2(1.0);
        vec2 v = vec2(1.0, 0.0);
        vec2 result = matrices[0] * v;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Transformed to typed GLSL_mul function
    assert 'GLSL_mul_mat2_vec2(matrices[0], v)' in opencl


def test_matrix_field_access_with_multiply(transformer):
    """Test matrix as struct field with multiplication."""
    ast_transformer, emitter = transformer
    glsl = """
    struct Transform {
        mat2 matrix;
    };

    void main() {
        Transform t;
        t.matrix = mat2(1.0);
        vec2 v = vec2(1.0, 0.0);
        vec2 result = t.matrix * v;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # Transformed to typed GLSL_mul function
    assert 'GLSL_mul_mat2_vec2(t.matrix, v)' in opencl


def test_transformation_shader_rotation(transformer):
    """Test real-world shader rotation transformation."""
    ast_transformer, emitter = transformer
    glsl = """
    void main() {
        vec2 uv = vec2(0.5, 0.5);
        float angle = 1.0;
        mat2 rotation = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
        vec2 rotatedUV = rotation * uv;
    }
    """
    parser = GLSLParser()
    ast = parser.parse(glsl)
    transformed = ast_transformer.transform(ast)
    opencl = emitter.emit(transformed)

    # mat2 full constructor uses GLSL_mat2(), multiplication uses typed GLSL_mul
    assert 'GLSL_cos' in opencl
    assert 'GLSL_sin' in opencl
    assert 'GLSL_mat2' in opencl
    assert 'GLSL_mul_mat2_vec2(rotation, uv)' in opencl
