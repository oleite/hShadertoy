import hou, json, os

# You can tweak this if you want a different default subnet name under /obj
SUBNET_BASENAME = "Untitled1"
COPNET_BASENAME = "copnet1"

def run(kwargs=None):
    # 1) ask for JSON
    start_dir = os.path.join(hou.expandString("$HOUDINI_USER_PREF_DIR"), "scripts")
    path = hou.ui.selectFile(
        start_directory=start_dir,
        title="Load COP Nodes (ignore saved paths)",
        pattern="*.json",
        chooser_mode=hou.fileChooserMode.Read
    )
    if not path:
        return

    with open(path, "r") as f:
        data = json.load(f)

    nodes_data = data.get("nodes", [])
    conns_data = data.get("connections", [])

    # 2) Make /obj/<UntitledX>/copnetY
    obj = hou.node("/obj")
    if obj is None:
        hou.ui.displayMessage("Cannot find /obj.", title="Error")
        return

    subnet = _create_unique_child(obj, "subnet", SUBNET_BASENAME)
    copnet = _create_unique_child(subnet, "copnet", COPNET_BASENAME)

    # 3) Create all nodes inside the COP network
    created = []
    errs = []
    name_to_node = {}

    cat = copnet.childTypeCategory()  # should be COPs category
    allowed = set(cat.nodeTypes().keys())

    for nd in nodes_data:
        try:
            node_type = nd["node_type"]
            node_name = nd["node_name"]
            parms_data = nd.get("parameters", {})

            if node_type not in allowed:
                raise hou.Error(f"Node type '{node_type}' is not valid in COPNET network.")

            # ensure node name unique under copnet
            new_name = _unique_name(copnet, node_name)
            node = copnet.createNode(node_type, node_name=new_name, run_init_scripts=False)

            # Try canonical parm data first (expressions/keyframes/ramps/multiparms)
            try:
                if parms_data:
                    node.setParmsFromData(parms_data)
            except Exception:
                _apply_parms_fallback(node, parms_data)

            created.append(node)
            name_to_node[node_name] = node  # map original name -> new node

        except Exception as e:
            errs.append(str(e))

    # 4) Rebuild connections (map by original node_name)
    made = 0
    for c in conns_data:
        try:
            # Original JSON likely stored absolute paths; we only care about names now.
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
            pass  # skip bad edges silently

    # 5) Layout (best effort)
    try: copnet.layoutChildren()
    except: pass

    msg = f"Created {len(created)} COP node(s) in {copnet.path()}; wired {made} connection(s)."
    if errs:
        msg += "\n\nSome items were skipped:\n- " + "\n- ".join(errs)
    hou.ui.displayMessage(msg, title="Load Complete")


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
