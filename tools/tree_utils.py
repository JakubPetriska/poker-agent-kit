def get_parent_action(node):
    if node.parent:
        return list(filter(lambda item: item[1] == node, node.parent.children.items()))[0][0]
    else:
        return None
