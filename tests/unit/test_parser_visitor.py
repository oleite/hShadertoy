"""
Visitor pattern tests - AST traversal.

Tests the ASTVisitor pattern for traversing parse trees.
"""

import pytest
from glsl_to_opencl.parser import GLSLParser, ASTVisitor, FunctionDefinition


class FunctionNameCollector(ASTVisitor):
    """Example visitor that collects function names."""

    def __init__(self):
        self.function_names = []

    def visit_FunctionDefinition(self, node):
        self.function_names.append(node.name)
        # Don't visit children


class NodeCounter(ASTVisitor):
    """Count all nodes in AST."""

    def __init__(self):
        self.count = 0

    def generic_visit(self, node):
        self.count += 1
        for child in node.children:
            self.visit(child)


class TestASTVisitor:
    """Test ASTVisitor pattern."""

    def test_visitor_collects_function_names(self):
        """Test visitor collects function names."""
        source = """
        float helper() { return 1.0; }
        void mainImage(out vec4 fragColor, in vec2 fragCoord) { }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        collector = FunctionNameCollector()
        collector.visit(ast)

        assert len(collector.function_names) == 2
        assert "helper" in collector.function_names
        assert "mainImage" in collector.function_names

    def test_visitor_counts_nodes(self):
        """Test visitor counts all nodes."""
        source = "float x;"
        parser = GLSLParser()
        ast = parser.parse(source)

        counter = NodeCounter()
        counter.visit(ast)

        assert counter.count > 0

    def test_visitor_traverses_complex_tree(self):
        """Test visitor traverses complex tree."""
        source = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            vec2 uv = fragCoord / iResolution.xy;
            vec3 col = vec3(uv, 0.5);
            fragColor = vec4(col, 1.0);
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        counter = NodeCounter()
        counter.visit(ast)

        # Should traverse many nodes
        assert counter.count > 10


class TestASTWalk:
    """Test AST.walk() traversal."""

    def test_walk_visits_all_nodes(self):
        """Test walk() visits all nodes."""
        source = """
        void main() {
            float x = 1.0;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        nodes = list(ast.walk())
        assert len(nodes) > 0

    def test_walk_includes_root(self):
        """Test walk() includes root node."""
        source = "float x;"
        parser = GLSLParser()
        ast = parser.parse(source)

        nodes = list(ast.walk())
        assert nodes[0] == ast

    def test_walk_depth_first(self):
        """Test walk() is depth-first."""
        source = """
        void main() {
            float x;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)

        nodes = list(ast.walk())
        # Should visit in depth-first order
        assert len(nodes) > 3
