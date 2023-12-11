import ast


class IllegaActionError(Exception):
    def __init__(self, message):
        super().__init__(message)


class LoopChecker(ast.NodeVisitor):
    def visit_While(self, node):
        raise IllegaActionError("Dangerous! Infinite or long loop detected: while loop")

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "range":
            if len(node.args) > 0 and isinstance(node.args[0], ast.Num):
                loop_count = node.args[0].n
                if loop_count > 1000:
                    raise IllegaActionError(
                        "Dangerous! Infinite or long loop detected: range() loop"
                    )
        self.generic_visit(node)


def check_code(code):
    tree = ast.parse(code)
    LoopChecker().visit(tree)
