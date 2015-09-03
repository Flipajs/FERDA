import random
import copy
import time


def pop(stack):
    return stack.pop()


def peek(stack):
    n = stack.pop()
    stack.append(n)
    return n


def contains(shortcuts, value):
    for sh in shortcuts:
        if sh.value == value:
            return True
    return False


def create_graph():
    random.seed(1)
    root = Node(0, "root", None)
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
                new = Node(depth, id, n)
                n.add_child(new)
                stack.append(new)
                # print "Adding a child to %s" % n.id
                chars = []
                for sh in new.shortcuts:
                    chars.append(sh.value)
                print "Adding a child to %s. ID is %s and shortcuts are %s. It will have %s children" % (n.id, new.id, chars, new.children_limit)
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


class Node():
    def __init__(self, depth, id, parent):
        self.depth = depth
        self.id = id
        self.parent = parent
        # choose some random "shortcuts"
        self.shortcuts = []
        for i in range(0, random.randint(1, 3)):
            rnd = random.randint(0, 20)
            k = Key(chr(rnd+1 + 64), self)
            self.shortcuts.append(k)
        self.shortcuts.sort()
        self.children = []
        self.children_limit = random.randint(0, 3)

    def add_child(self, node):
        self.children.append(node)

    def children_count(self):
        return len(self.children)


class Key():
    def __init__(self, value, parent):
        self.value = value
        self.parent = parent


def search_graph(parent):
    # temporary storage for all children's shortcuts
    short = []
    for child in parent.children:

        seen = []
        for sh in child.shortcuts:
            if sh.value not in seen:
                seen.append(sh.value)
            else:
                print "Conflict! The key %s can be used only once in %s" % (sh.value, child.id)
                child.shortcuts.remove(sh)

        if len(child.children) > 0:
            # load child's children
            shortcuts = search_graph(child)
            # check all shortcuts
            for sh in shortcuts:
                # check if it collides with any key in child.shortcuts
                if contains(child.shortcuts, sh.value):
                    print "Conflict! The key %s from %s can't be used also in %s" % (sh.value, child.id, sh.parent.id)
                # add it to short
                if not contains(short, sh):
                    short.append(sh)

        for sh in child.shortcuts:
            if not contains(short, sh):
                short.append(sh)

    # short.sort()
    return short


t = time.time()
root = create_graph()
print "Time taken to create graph: %s" % (time.time() - t)

t = time.time()
search_graph(root)
print "Time taken to check graph: %s" % (time.time() - t)