__author__ = 'filip@naiser.cz'
import math
from matplotlib.mlab import normpdf


def evolve_score_functions(ant, region, params, score_functions, expression):
    """Evolves given score functions and returns expression result.

    Args:
        ant: class Ant from ant.py
        region: dict of mser region parameters
        params: class Params from experiment_parameters.py
        score_functions: list of function names(str)
        expression: str with expression to be evaluated.

    Returns:
        float, value as an result of expression

    Raises:
        NameError: An error occurred during calling some of the score_functions
            or during evaluation of expression
        SyntaxError: In case when there si syntax problem in expression

    Example:
        ...def f1(a, r): ... def f2(a, r): ...
        expression = '2*f2+f1' or '2*$1+$0

        evolve_score_functions(a, r, ['f1', 'f2'], expression)
    """

    ns = get_globals_namespace()

    i = 0
    for s_fce in score_functions:
        s = eval(s_fce+'(region, ant, params)')
        #in case of expr = 2*f1 + f2
        expression = expression.replace(s_fce, str(s))

        #in case of expr = 2*$0 + $f2
        expression = expression.replace('$'+str(i), str(s))

        i += 1

    return eval(expression, ns)


def get_globals_namespace():
    """ returns dict of global context for safe evaluation of eval function.

    Returns:
        dict of global context
    """


    #allowing to use math
    ns = vars(math).copy()

    #adding score funtcions
    ns['f1'] = f1
    ns['f2'] = f2

    #this approach should prevent injections in eval function
    ns['__builtins__'] = None


def distance_score(ant, region, params):
    """ Returns value in the range <0;1> based on distance between ant and region.

    Args:
        ant: class Ant from ant.py
        region: dict
        params: class Params from experiment_parameters.py

    Returns:
        float from interval <0;1>
    """

    u = 0  # mean
    s = params.avg_ant_axis_a   # standard deviation
    max_val = normpdf(u, u, s)

    x = ant.state.position.x - region['cx']
    y = ant.state.position.y - region['cy']

    d = math.sqrt(x**2 + y**2)

    val = normpdf(d, u, s) / max_val  # division arranges normalization

    return val


def f1(ant, reg):
    return ant+reg


def f2(ant, reg):
    return 3.1


def my_max(a, b):
    if a > b:
        return a

    return b