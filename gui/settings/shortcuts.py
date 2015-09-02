import random
import copy
import time

class Node():
    def __init__(self, depth, id):
        self.depth = depth
        self.id = id
        # choose some random "shortcuts" (represented by ints)
        self.shortcuts = []
        for i in range(0, random.randint(1, 3)):
            rnd = random.randint(0, 20)
            while rnd in self.shortcuts:
                rnd = random.randint(0, 20)
            self.shortcuts.append(rnd)
        self.shortcuts.sort()
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


def create_graph():
    random.seed(1)
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
                print "Adding a child to %s. ID is %s and shortcuts are %s. It will have %s children" % (n.id, new.id, new.shortcuts, new.children_limit)
                n = new
                depth += 1
                id += 1
            else:
                # close this node
                try:
                    pop(stack)
                    n = peek(stack)
                except IndexError:
                    break

                depth -= 1
    print "Total children created: %s" % id
    return root


def search_graph(parent):
    # temporary storage for all children's shortcuts
    short = []
    for child in parent.children:

        if len(child.children) > 0:
            # load child's children
            shortcuts = search_graph(child)
            # check all shortcuts
            for sh in shortcuts:
                # check if it collides with any key in child.shortcuts
                if sh in child.shortcuts:
                    print "Conflict! Key %s from %s" % (sh, child.id)
                # add it to short
                if sh not in short:
                    short.append(sh)

        for sh in child.shortcuts:
            if sh not in short:
                short.append(sh)

    # return all keys used in parent's children
    short.sort()
    # print "Children of %s have the following shortcuts: %s" % (parent.id, short)
    return short


t = time.time()
root = create_graph()
print "Time taken to create graph: %s" % (time.time() - t)

t = time.time()
search_graph(root)
print "Time taken to read graph: %s" % (time.time() - t)

