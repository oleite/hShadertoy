import sys, hou, json, os, argparse

# example usage:
# hython template_load_headless.py --cook none "resources/examples/gameOfLife/gameOfLife_HDA.json"

# You can tweak this if you want a different default subnet name under /obj
SUBNET_BASENAME = "Untitled1"
COPNET_BASENAME = "copnet1"

def run_headless(json_path: str, cook: str = "none"):
    """Non-UI entrypoint for hython/CLI testing."""
    import json, hou
    # mimic the guts of run() but skip file chooser & message dialog
    with open(json_path, "r") as f:
        data = json.load(f)

    nodes_data = data.get("nodes", [])
    conns_data = data.get("connections", [])

    obj = hou.node("/obj")
    if obj is None:
        raise RuntimeError("Cannot find /obj.")

    subnet = _create_unique_child(obj, "subnet", SUBNET_BASENAME)
    copnet = _create_unique_child(subnet, "copnet", COPNET_BASENAME)

    created = []
    errs = []
    name_to_node = {}

    cat = copnet.childTypeCategory()
    allowed = set(cat.nodeTypes().keys())

    for nd in nodes_data:
        try:
            node_type = nd["node_type"]
            node_name = nd["node_name"]
            parms_data = nd.get("parameters", {})

            if node_type not in allowed:
                raise hou.Error(f"Node type '{node_type}' is not valid in COPNET network.")

            new_name = _unique_name(copnet, node_name)
            node = copnet.createNode(node_type, node_name=new_name, run_init_scripts=False)

            try:
                if parms_data:
                    node.setParmsFromData(parms_data)
            except Exception:
                _apply_parms_fallback(node, parms_data)

            created.append(node)
            name_to_node[node_name] = node
        except Exception as e:
            errs.append(str(e))

    made = 0
    for c in conns_data:
        try:
            src_name = _basename(c.get("src", "")) or c.get("src_name")
            dst_name = _basename(c.get("dst", "")) or c.get("dst_name")
            if not src_name or not dst_name:
                continue
            src = name_to_node.get(src_name)
            dst = name_to_node.get(dst_name)
            if not src or not dst:
                continue
            dst_input_index = int(c.get("dst_input_index", 0))
            src_output_index = int(c.get("src_output_index", 0))
            dst.setInput(dst_input_index, src, src_output_index)
            made += 1
        except Exception:
            pass

    msg = f"Created {len(created)} COP node(s) in {copnet.path()}; wired {made} connection(s)."
    if errs:
        msg += "\nSome items were skipped:\n- " + "\n- ".join(errs)
    print(msg)

    if cook not in ("none", "opencl", "all"):
        cook = "none"
        
    if cook != "none":
        def _should_cook(n: hou.Node) -> bool:
            return (cook == "all") or (n.type().name() == "opencl")

        for n in created:
            if not _should_cook(n):
                continue
            try:
                # Force a cook; this triggers OpenCL compile on OpenCL COPs
                # nodelang = n.cookCodeLanguage()
                print(f"Cooking {n.path()}")
                n.cook(force=True)
                
                # Also surface any node-side error text if present
                try:
                    msgs = n.errors()
                    if msgs:
                        errs.append(f"[cook] {n.path()}: " + " | ".join(msgs))
                except Exception:
                    pass
            except Exception as e:
                # Collect compile/cook failures
                errtxt = str(e)
                try:
                    msgs = n.errors()
                    if msgs:
                        errtxt += " | " + " | ".join(msgs)
                except Exception:
                    pass
                errs.append(f"[cook] {n.path()}: {errtxt}")
            





# ---------- helpers ----------
def _create_unique_child(parent, type_name, base_name):
    """
    Always create a NEW child node of given type under 'parent'.
    Use 'base_name' if available; otherwise append 1,2,3... until free.
    """
    name = base_name if parent.node(base_name) is None else _unique_name(parent, base_name)
    return parent.createNode(type_name, node_name=name)

def _ensure_unique_child(parent, type_name, base_name):
    """
    Get or create a child node of the given type under 'parent' with a unique name
    based on base_name (base_name, base_name1, base_name2, ...).
    If a node with exact base_name already exists and has the correct type, reuse it.
    Otherwise, create a new one with enumerated suffix.
    """
    n = parent.node(base_name)
    if n is not None:
        if n.type().name() == type_name:
            return n
        # name clash with different type, fall through to create enumerated
    name = _unique_name(parent, base_name)
    return parent.createNode(type_name, node_name=name)

def _unique_name(parent, desired_name):
    """
    Return 'desired_name' if available; otherwise append 1,2,3... until free.
    """
    if parent.node(desired_name) is None:
        return desired_name
    i = 1
    while True:
        name = f"{desired_name}{i}"
        if parent.node(name) is None:
            return name
        i += 1

def _basename(p):
    p = (p or "").strip("/")
    return p.split("/")[-1] if p else ""

def _apply_parms_fallback(node, parms_data):
    """
    Lenient per-parm setter so slightly non-recipe data still applies.
    Prefer setParmsFromData for canonical shapes.
    """
    if not parms_data:
        return
    for name, val in parms_data.items():
        parm = node.parm(name)
        if parm is None:
            continue
        try:
            # Expression dict: {"expression": "...", "language": "Hscript|Python"}
            if isinstance(val, dict) and "expression" in val:
                lang = val.get("language", "Hscript")
                expr_lang = hou.exprLanguage.Python if str(lang).lower().startswith("py") else hou.exprLanguage.Hscript
                parm.setExpression(val["expression"], expr_lang)
                continue
            if isinstance(val, (int, float, str)):
                parm.set(val)
                continue
            if isinstance(val, list):
                pt = parm.tuple()
                if pt and pt.size() == len(val):
                    pt.set(val)
                else:
                    parm.set(val[0] if val else parm.eval())
                continue
            parm.set(val)
        except Exception:
            pass

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("template", help="Path to template JSON")
    ap.add_argument("--cook", choices=["none", "opencl", "all"], default="none",
                    help="Force-cook nodes after creation (to compile OpenCL)")
    args = ap.parse_args()
    run_headless(args.template, cook=args.cook)
