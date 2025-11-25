"""
Visitor pattern for AST traversal.
"""

from .ast_nodes import ASTNode


class ASTVisitor:
    """
    Visitor pattern for traversing AST.

    Subclass and override visit_* methods to implement custom traversal logic.

    Example:
        class FunctionCollector(ASTVisitor):
            def __init__(self):
                self.functions = []

            def visit_FunctionDefinition(self, node):
                self.functions.append(node.name)
                self.generic_visit(node)  # Continue traversal

        collector = FunctionCollector()
        collector.visit(ast)
        print(collector.functions)  # ['mainImage', 'helper', ...]
    """

    def visit(self, node: ASTNode):
        """
        Visit a node.

        Dispatches to visit_<NodeClass> method if it exists,
        otherwise calls generic_visit().
        """
        method_name = f"visit_{node.__class__.__name__}"
        visitor_method = getattr(self, method_name, self.generic_visit)
        return visitor_method(node)

    def generic_visit(self, node: ASTNode):
        """
        Default visitor - traverses all children.

        Override this to change default traversal behavior.
        """
        for child in node.children:
            self.visit(child)
