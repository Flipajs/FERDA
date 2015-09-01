import random
import copy

class Node():
    def __init__(self, depth, id):
        self.depth = depth
        self.id = id
        self.children = []
        self.children_limit = random.randint(0, 3)

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

root = Node(0, "root")
n = copy.copy(root)
stack = []
stack.append(n)

depth = 0
id = 1

while len(stack) > 0:
    # print depth, id
    # if the open node doesn't have 2 children yet
    # print depth, len(stack), id
    if depth > 4:
        # print "We do not have to go deeper than %s" % depth
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