import random
import copy
import time

class Node():
    def __init__(self, depth, id):
        self.depth = depth
        self.id = id
        # choose some random "shortcuts" (represented by ints)
        self.shortcuts = []
        for i in range(0, random.randint(2, 5)):
            self.shortcuts.append(random.randint(0, 20))
        self.children = []
        self.children_limit = random.randint(0, 3)
        self.status = "fresh"

    def add_child(self, node):
        self.children.append(node)

    def children_count(self):
        return len(self.children)

def pop(stack):
    return stack.pop()


def peek(stack):
    n = stack.pop()
    stack.append(n)
    return n


def create_graph():
    root = Node(0, "root")
    n = copy.copy(root)
    stack = []
    stack.append(n)

    depth = 0
    id = 1

    while len(stack) > 0:
        # do not go deeper than 4
        if depth > 4:
            pop(stack)
            n = peek(stack)
            depth -= 1
        else:
            if n.children_count() < n.children_limit:
                # add a child
                new = Node(depth, id)
                n.add_child(new)
                stack.append(new)
                # print "Adding a child to %s" % n.id
                n = new
                depth += 1
                id += 1
            else:
                # close this node
                # print "%s already has %s children" % (n.id, n.children_count())

                try:
                    pop(stack)
                    n = peek(stack)
                except IndexError:
                    break

                depth -= 1
    print "Total children created: %s" % id
    return root


def search_graph(first_node):
    first_node.status = "open"
    for child in first_node.children:
        if len(child.children) > 0:
            ok = True
            shortcuts = search_graph(child)
            for sh in shortcuts:
                if sh in child.shortcuts:
                    ok = False
                    print "Conflict! Key %s from %s" % (sh, child.id)
                else:
                    child.shortcuts.append(sh)
            return child.shortcuts
        else:
            print "end found!"
            return child.shortcuts


t = time.time()
root = create_graph()
print "Time taken to create graph: %s" % (time.time() - t)

t = time.time()
search_graph(root)
print "Time taken to read graph: %s" % (time.time() - t)

