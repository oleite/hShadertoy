"""
Unit tests for AST transformation infrastructure.

Tests the core infrastructure for transforming GLSL AST to OpenCL IR:
- TransformedASTNode creation and structure
- ASTTransformer visitor pattern
- CodeEmitter basic functionality
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import (
    SymbolTable,
    TypeChecker,
    create_builtin_symbol_table,
)
from src.glsl_to_opencl.analyzer.type_checker import TYPE_NAME_MAP
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.transformer import transformed_ast as IR
from src.glsl_to_opencl.transformer.code_emitter import CodeEmitter


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def parser():
    """Create GLSL parser."""
    return GLSLParser()


@pytest.fixture
def symbol_table():
    """Create symbol table with built-in functions."""
    return create_builtin_symbol_table()


@pytest.fixture
def type_checker(symbol_table):
    """Create type checker."""
    return TypeChecker(symbol_table)


@pytest.fixture
def transformer(type_checker):
    """Create AST transformer."""
    return ASTTransformer(type_checker)


@pytest.fixture
def emitter():
    """Create code emitter."""
    return CodeEmitter()


# ============================================================================
# TransformedNode Tests (10 tests)
# ============================================================================

def test_float_literal_creation():
    """Test creating FloatLiteral node."""
    node = IR.FloatLiteral(
        value='1.0f',
        glsl_type=TYPE_NAME_MAP['float'],
        source_location=(0, 0)
    )
    assert node.value == '1.0f'
    assert node.glsl_type == TYPE_NAME_MAP['float']


def test_float_literal_requires_suffix():
    """Test FloatLiteral validates 'f' suffix."""
    with pytest.raises(ValueError, match="must end with 'f'"):
        IR.FloatLiteral(value='1.0', glsl_type=TYPE_NAME_MAP['float'])


def test_int_literal_creation():
    """Test creating IntLiteral node."""
    node = IR.IntLiteral(
        value='42',
        glsl_type=TYPE_NAME_MAP['int'],
        source_location=(0, 0)
    )
    assert node.value == '42'


def test_bool_literal_creation():
    """Test creating BoolLiteral node."""
    node = IR.BoolLiteral(
        value=True,
        glsl_type=TYPE_NAME_MAP['bool'],
        source_location=(0, 0)
    )
    assert node.value is True


def test_identifier_creation():
    """Test creating Identifier node."""
    node = IR.Identifier(
        name='myVar',
        glsl_type=TYPE_NAME_MAP['float'],
        source_location=(0, 0)
    )
    assert node.name == 'myVar'


def test_binary_op_creation():
    """Test creating BinaryOp node."""
    left = IR.FloatLiteral(value='1.0f')
    right = IR.FloatLiteral(value='2.0f')
    node = IR.BinaryOp(
        operator='+',
        left=left,
        right=right,
        source_location=(0, 0)
    )
    assert node.operator == '+'
    assert node.left == left
    assert node.right == right


def test_call_expression_creation():
    """Test creating CallExpression node."""
    arg = IR.FloatLiteral(value='1.0f')
    node = IR.CallExpression(
        function='GLSL_sin',
        arguments=[arg],
        source_location=(0, 0)
    )
    assert node.function == 'GLSL_sin'
    assert len(node.arguments) == 1


def test_type_constructor_creation():
    """Test creating TypeConstructor node."""
    arg1 = IR.FloatLiteral(value='1.0f')
    arg2 = IR.FloatLiteral(value='2.0f')
    node = IR.TypeConstructor(
        type_name='float2',
        arguments=[arg1, arg2],
        source_location=(0, 0)
    )
    assert node.type_name == 'float2'
    assert len(node.arguments) == 2


def test_declaration_creation():
    """Test creating Declaration node."""
    init = IR.FloatLiteral(value='1.0f')
    node = IR.Declaration(
        type_name='float',
        name='x',
        initializer=init,
        source_location=(0, 0)
    )
    assert node.type_name == 'float'
    assert node.name == 'x'
    assert node.initializer == init


def test_function_definition_creation():
    """Test creating FunctionDefinition node."""
    param = IR.Parameter(type_name='float', name='x', qualifiers=[])
    body = IR.CompoundStatement(statements=[])
    node = IR.FunctionDefinition(
        return_type='float',
        name='compute',
        parameters=[param],
        body=body,
        source_location=(0, 0)
    )
    assert node.return_type == 'float'
    assert node.name == 'compute'
    assert len(node.parameters) == 1


# ============================================================================
# CodeEmitter Tests (10 tests)
# ============================================================================

def test_emit_float_literal(emitter):
    """Test emitting float literal."""
    node = IR.FloatLiteral(value='3.14f')
    result = emitter.emit(node)
    assert result == '3.14f'


def test_emit_int_literal(emitter):
    """Test emitting integer literal."""
    node = IR.IntLiteral(value='42')
    result = emitter.emit(node)
    assert result == '42'


def test_emit_bool_literal(emitter):
    """Test emitting boolean literal."""
    node_true = IR.BoolLiteral(value=True)
    node_false = IR.BoolLiteral(value=False)
    assert emitter.emit(node_true) == 'true'
    assert emitter.emit(node_false) == 'false'


def test_emit_identifier(emitter):
    """Test emitting identifier."""
    node = IR.Identifier(name='myVariable')
    result = emitter.emit(node)
    assert result == 'myVariable'


def test_emit_binary_op(emitter):
    """Test emitting binary operation."""
    left = IR.FloatLiteral(value='1.0f')
    right = IR.FloatLiteral(value='2.0f')
    node = IR.BinaryOp(operator='+', left=left, right=right)
    result = emitter.emit(node)
    assert result == '1.0f + 2.0f'


def test_emit_call_expression(emitter):
    """Test emitting function call."""
    arg = IR.FloatLiteral(value='1.0f')
    node = IR.CallExpression(function='GLSL_sin', arguments=[arg])
    result = emitter.emit(node)
    assert result == 'GLSL_sin(1.0f)'


def test_emit_type_constructor(emitter):
    """Test emitting type constructor."""
    arg1 = IR.FloatLiteral(value='1.0f')
    arg2 = IR.FloatLiteral(value='2.0f')
    node = IR.TypeConstructor(type_name='float2', arguments=[arg1, arg2])
    result = emitter.emit(node)
    assert result == '(float2)(1.0f, 2.0f)'


def test_emit_declaration(emitter):
    """Test emitting variable declaration."""
    init = IR.FloatLiteral(value='1.0f')
    node = IR.Declaration(type_name='float', name='x', initializer=init)
    result = emitter.emit(node)
    assert 'float x = 1.0f;' in result


def test_emit_return_statement(emitter):
    """Test emitting return statement."""
    value = IR.FloatLiteral(value='1.0f')
    node = IR.ReturnStatement(value=value)
    result = emitter.emit(node)
    assert 'return 1.0f;' in result


def test_emit_function_definition(emitter):
    """Test emitting function definition."""
    param = IR.Parameter(type_name='float', name='x', qualifiers=[])
    ret_stmt = IR.ReturnStatement(value=IR.Identifier(name='x'))
    body = IR.CompoundStatement(statements=[ret_stmt])
    node = IR.FunctionDefinition(
        return_type='float',
        name='identity',
        parameters=[param],
        body=body
    )
    result = emitter.emit(node)

    assert 'float identity' in result
    assert 'float x' in result
    assert 'return x;' in result
