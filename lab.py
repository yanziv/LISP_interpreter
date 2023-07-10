import sys
import doctest
from typing import Any
from webbrowser import get

sys.setrecursionlimit(20_000)

# NO ADDITIONAL IMPORTS!

#############################
# Scheme-related Exceptions #
#############################


class SchemeError(Exception):
    """
    A type of exception to be raised if there is an error with a Scheme
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass


class SchemeSyntaxError(SchemeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """

    pass


class SchemeNameError(SchemeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class SchemeEvaluationError(SchemeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SchemeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(value):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Scheme
                      expression
    """
    # need to deal with comments
    tokens = ["(", ")", "+", "*", "/"]  # '-' is not included bc it serves as dashes
    ele = ""
    result = []
    ignore = False  # keep track of comments
    for i, char in enumerate(source):
        if char == ";":  # start of a comment
            ignore = True
            continue
        if char == "\n":  # new line
            ignore = False
            if not ignore and ele != "":
                # print('ele0',ele =='',ele)
                result.append(ele)
            ele = ""
            continue

        if not ignore:
            if char == " ":
                if ele != "":
                    # print('ele1',ele =='',ele)
                    result.append(ele)
                ele = ""
            if char in tokens:
                if len(ele) > 0:
                    # print('ele2',ele =='')
                    result.append(ele)
                    ele = ""
                result.append(char)
            else:
                if char != " ":
                    ele += char
                    # print('ele3',ele =='')
            if ele == "":
                continue

    if len(ele) > 0:
        result.append(ele)
    return result


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    # print('tokens',tokens)
    def parse_expression(index):
        token = tokens[index]

        # base case: if it's not a parenthesis
        if token not in ["(", ")"]:
            return number_or_symbol(token), index + 1

        # recursive case
        # print('index',index)
        if token == "(":
            result = []

            # because we hit '(', so we increment index here
            index += 1
            while index < len(tokens) and tokens[index] != ")":
                temp, index = parse_expression(index)
                result.append(temp)

            return result, index + 1

        elif token == ")":
            raise SchemeSyntaxError("invalid input!")

    parsed_expression, next_index = parse_expression(0)

    if next_index != len(tokens):  # never reached the end of list, invalid input
        raise SchemeSyntaxError("invalid input!")

    return parsed_expression


######################
# Built-in Functions #
######################
def mul(args):
    """
    Given a list of numbers, returns a result of multiplying all nums together.
    """
    val = 1
    # print("mul args:", args)
    for num in args:
        val *= num
    return val


def div(args):
    """
    Given a list of numbers, returns a result of dividing the first number by 
    the next till reach the end of list.
    """
    first = args[0]
    for i in range(1, len(args)):
        first = first / args[i]
    return first


def equal(args):
    """
    Given a list of numbers, function should evaluate to true if all of 
    its arguments are equal to each other.
    """
    prev_arg = args[0]
    for i in range(1, len(args)):
        if prev_arg != args[i]:
            return False
        prev_arg = args[i]
    return True


def bigger_than(args):
    """
    Function should evaluate to true if its arguments are in decreasing order.
    """
    prev_arg = args[0]
    for i in range(1, len(args)):
        if prev_arg <= args[i]:
            return False
        prev_arg = args[i]
    return True


def bigger_or_equal(args):
    """
    Function should evaluate to true if its arguments are in nonincreasing order.
    """
    prev_arg = args[0]
    for i in range(1, len(args)):
        if prev_arg < args[i]:
            return False
        prev_arg = args[i]
    return True


def smaller_than(args):
    """
    Function should evaluate to true if its arguments are in increasing order.
    """
    prev_arg = args[0]
    for i in range(1, len(args)):
        if prev_arg >= args[i]:
            return False
        prev_arg = args[i]
    return True


def smaller_or_equal(args):
    """
    Function should evaluate to true if its arguments are in nondecreasing order.
    """
    prev_arg = args[0]
    for i in range(1, len(args)):
        if prev_arg > args[i]:
            return False
        prev_arg = args[i]
    return True


def Not(args):
    """
    Function takes a single argument and should evaluate to false if its argument
    is true and true if its argument is false.
    """
    if len(args) > 1 or len(args) == 0:
        raise SchemeEvaluationError

    if args[0]:
        return False
    return True


def cons(args):
    """
    This function constructs new objects and is used to make ordered pairs.
    """
    if len(args) != 2:
        raise SchemeEvaluationError
    return Pair(args[0], args[1])


def get_car(args):
    """
    This function takes a cons cell (an instance of your Pair class) as argument
    and should return the first element in the pair.
    """
    if len(args) != 1 or not isinstance(args[0], Pair):
        raise SchemeEvaluationError
    return args[0].car


def get_cdr(args):
    """
    This function takes a cons cell as argument and return
    the second element in the pair. 
    """
    if len(args) != 1 or not isinstance(args[0], Pair):
        raise SchemeEvaluationError
    return args[0].cdr


def generate_linked_lists(args):
    """
    Builtin function that constructs a linked list using the input args.
    """
    if len(args) == 0:
        return None
    return Pair(args[0], generate_linked_lists(args[1:]))


# convenience methods
def islinkedlist(obj):
    """
    This function takes an arbitrary object as input, and it should
    return #t if that object is a linked list, and #f otherwise.
    """
    # print('obj1',obj)
    if obj[0] == None:  # nil is a valid list
        return True
    # edge case for (list? 7)
    if not isinstance(obj[0], Pair):
        return False
    curr = obj[0].cdr
    while curr is not None:
        if not isinstance(curr, Pair):
            return False
        curr = curr.cdr
        # print('updated curr',curr)
    return True


def length_list(alist):
    """
    take a list as argument and should return the length of that list.
    When called on any object that is not a linked list, it should raise
    a SchemeEvaluationError.
    """
    if not islinkedlist(alist):  # not a linked list
        raise SchemeEvaluationError("Not a linked list!")

    if alist[0] == None:
        return 0
    curr = alist[0].cdr
    # print('curr',curr)

    return length_list([curr]) + 1


def list_ref(args):
    """
    This function takes a list and a nonnegative index, and it should return
    the element at the given index in the given list.
    """
    # print("args", args)
    alist = args[0]
    idx = args[1]
    # if LIST is a cons cell and not a list
    if isinstance(alist, Pair) and not islinkedlist([alist]):
        if idx == 0:
            return alist.car
        else:
            raise SchemeEvaluationError("index out of range!")

    if alist is None:
        raise SchemeEvaluationError("list item not accessible!")
    if idx == 0:
        return alist.car

    return list_ref([alist.cdr, idx - 1])


def append(lists):
    """
    The function takes an arbitrary number of lists as arguments and
    should return a new list representing the concatenation of these lists.
    Note that this append is different from Python's, in that this should
    not mutate any of its arguments.
    """
    if len(lists) == 0:
        return None
    elif len(lists) == 1:
        result = lists[0]
        return result

    new_list = None
    final_list = None
    for alist in lists:
        if not islinkedlist([alist]):
            raise SchemeEvaluationError("input not a linked list!")
        curr = alist
        while islinkedlist([curr]) and curr is not None:
            temp = Pair(curr.car, None)
            if new_list is None:
                new_list = temp
                final_list = new_list
            else:
                new_list.cdr = temp
                new_list = new_list.cdr
            curr = curr.cdr

    return final_list


def map_func(args):
    """
    takes a function and a list as arguments, and it returns a new list containing
    the results of applying the given function to each element of the given list.
    For example, (map (lambda (x) (* 2 x)) (list 1 2 3)) should produce the list (2 4 6).
    """
    func = args[0]
    alist = args[1]

    length = length_list([alist])
    # print('length',length)
    new_list = None
    final_list = None
    for i in range(length):
        elem = list_ref([alist, i])
        # print('elem',elem)
        temp = Pair(func([elem]), None)
        if new_list is None:
            new_list = temp
            final_list = new_list
        else:
            new_list.cdr = temp
            new_list = new_list.cdr

    return final_list


def filter_func(args):
    """
    takes a function and a list as arguments, and it returns a new list containing
    only the elements of the given list for which the given function returns true.
    """
    func = args[0]
    alist = args[1]

    length = length_list([alist])
    new_list = None
    final_list = None
    for i in range(length):
        elem = list_ref([alist, i])
        if func([elem]):
            temp = Pair(elem, None)
            if new_list is None:
                new_list = temp
                final_list = new_list
            else:
                new_list.cdr = temp
                new_list = new_list.cdr

    return final_list


def reduce_func(args):
    """
    takes a function, a list, and an initial value as inputs. It produces its output
    by successively applying the given function to the elements in the list, maintaining
    an intermediate result along the way.
    """
    func = args[0]
    alist = args[1]
    init_val = args[2]

    length = length_list([alist])
    for i in range(length):
        elem = list_ref([alist, i])
        init_val = func([init_val, elem])

    return init_val


def begin(args):
    """
    This function simply returns its last argument. For example,
    (begin (define x 7) (define y 8) (- x y)) should evaluate to -1.
    """
    return args[-1]


def evaluate_file(filename, frame=None):
    """
    This function should take a single argument (a string containing
    the name of a file to be evaluated) and an optional argument (the
    frame in which to evaluate the expression), and it should return the
    result of evaluating the expression contained in the file.
    """

    file = open(filename).read()
    return evaluate(parse(tokenize(file)), frame)


scheme_builtins = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": mul,
    "/": div,
    "equal?": equal,
    ">": bigger_than,
    ">=": bigger_or_equal,
    "<": smaller_than,
    "<=": smaller_or_equal,
    "not": Not,
    "#t": True,
    "#f": False,
    "car": get_car,
    "cdr": get_cdr,
    "list": generate_linked_lists,
    "cons": cons,
    "nil": None,
    "list?": islinkedlist,
    "length": length_list,
    "list-ref": list_ref,
    "append": append,
    "map": map_func,
    "filter": filter_func,
    "reduce": reduce_func,
    "begin": begin,
}
##############
# Evaluation #
##############


def define_helper(tree, frame=None):
    """
    Helper function for when keyword is 'define'.
    """
    if isinstance(tree[1], str):
        var = tree[1]
        val = evaluate_helper(tree[2], frame)
        frame.bindings[var] = val
        return val
    elif isinstance(tree[1], list):
        func_name = tree[1][0]
        params = tree[1][1:]
        body = tree[2]
        new_func = Function(params, body, frame)
        frame.bindings[func_name] = new_func
        return new_func


def if_helper(tree, frame=None):
    """
    Helper function for when keywrod is 'if'.
    """
    pred = evaluate_helper(tree[1], frame)
    if pred:
        return evaluate_helper(tree[2], frame)
    return evaluate_helper(tree[3], frame)


def let_helper(tree, frame=None):
    """
    Helper function for when keywrod is 'let'.
    """
    var_val_pairs = tree[1]
    body = tree[2]
    new_frame = Frame(frame)
    for pair in var_val_pairs:
        new_frame.bindings[pair[0]] = evaluate(pair[1], frame)
    # print("new_frame_bidings", new_frame.bindings)
    return evaluate(body, new_frame)


def del_helper(tree, frame=None):
    """
    Helper function for when keywrod is 'del'.
    """
    var = tree[1]
    if var not in frame.bindings:
        raise SchemeNameError("var is not bound locally!")
    else:
        return frame.bindings.pop(var)


def setbang_helper(tree, frame):
    """
    Helper function for when keywrod is 'set!'.
    """
    var = tree[1]
    exp = evaluate(tree[2], frame)

    frame.set_bang(var, exp)
    return exp


def lambda_helper(tree, frame):
    """
    Helper function for when keywrod is 'lambda'.
    """
    params = tree[1]
    body = tree[2]
    return Function(params, body, frame)


def and_helper(tree, frame):
    """
    Helper function for when keywrod is 'and'.
    """
    for sub_exp in tree[1:]:
        bool_val = evaluate_helper(sub_exp, frame)
        if not bool_val:
            return False
    return True


def or_helper(tree, frame):
    """
    Helper function for when keywrod is 'or'.
    """
    for sub_exp in tree[1:]:
        bool_val = evaluate_helper(sub_exp, frame)
        if bool_val:
            return True
    return False


def else_helper(keyword, tree, frame):
    """
    Helper function for when keyword doesn't match any of the available options.
    """
    func = evaluate_helper(keyword, frame)
    if not callable(func):
        raise SchemeEvaluationError
    num_list = []
    for elem in tree[1:]:
        num_list.append(evaluate_helper(elem, frame))

    return func(num_list)


def evaluate_helper(tree, frame=None):
    """
    A functions that takes the same arguments as evaluate() but
    returns a tuple with 2 elements: the result of the evaluation and 
    the frame in which the expression was evaluated.

    If a frame is given, the expression should be evaluated in that frame.
    If no frame is given, you should make a brand new frame and evaluate 
    the expression in that frame.
    """
    # print('tree',tree)
    if frame is None:
        frame = Frame()
        frame.parent_frame = Frame()
        frame.parent_frame.bindings = scheme_builtins

    # base case: tree is either a number, an operator, or a var
    if isinstance(tree, (int, float)):  # a number
        return tree
    if isinstance(tree, str):
        return frame.get_val(tree)

    # recursive case
    if isinstance(tree, list):
        if len(tree) == 0:
            raise SchemeEvaluationError("empty tree.")
        keyword = tree[0]
        if keyword == "define":
            return define_helper(tree, frame)

        elif keyword == "lambda":
            return lambda_helper(tree, frame)

        elif keyword == "if":
            return if_helper(tree, frame)

        # boolean combinators
        elif keyword == "and":
            return and_helper(tree, frame)

        elif keyword == "or":
            return or_helper(tree, frame)

        # variable-binding manipulation
        elif keyword == "del":
            return del_helper(tree, frame)

        elif keyword == "let":
            return let_helper(tree, frame)

        elif keyword == "set!":
            return setbang_helper(tree, frame)

        else:
            return else_helper(keyword, tree, frame)


def evaluate(tree, frame=None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """

    result = evaluate_helper(tree, frame)
    return result


def result_and_frame(tree, frame=None):
    """
    A functions that takes the same arguments as evaluate() but
    returns a tuple with 2 elements: the result of the evaluation and 
    the frame in which the expression was evaluated.
    """

    if frame is None:
        frame = Frame()
        frame.parent_frame = Frame()
        frame.parent_frame.bindings = scheme_builtins

    eval = evaluate_helper(tree, frame)
    return eval, frame


def repl(verbose=False):
    """
    Read in a single line of user input, evaluate the expression, and print 
    out the result. Repeat until user inputs "QUIT"
    
    Arguments:
        verbose: optional argument, if True will display tokens and parsed
            expression in addition to more detailed error output.
    """

    # updated REPL
    import traceback

    _, frame = result_and_frame(["+"])  # make a global frame
    command_line_args = sys.argv
    # print("command", command_line_args)
    if len(command_line_args) > 0:
        for file in command_line_args:
            if file != "lab.py":
                evaluate_file(file, frame)

    while True:
        input_str = input("in> ")
        if input_str == "QUIT":
            return
        try:
            token_list = tokenize(input_str)
            if verbose:
                print("tokens>", token_list)
            expression = parse(token_list)
            if verbose:
                print("expression>", expression)
            output, frame = result_and_frame(expression, frame)
            print("  out>", output)
        except SchemeError as e:
            if verbose:
                traceback.print_tb(e.__traceback__)
            print("Error>", repr(e))


class Pair:
    """
    Pair class.
    """

    def __init__(self, car, cdr):
        # car represents the 1st element in pair
        # cdr represents the 2nd element in pair
        self.car = car
        self.cdr = cdr


class Frame:
    """
    Frame class.
    """

    def __init__(self, parent_frame=None):
        # a frame might not have a parent frame
        self.parent_frame = parent_frame
        self.bindings = {}

    def get_val(self, name):
        if name in self.bindings:
            return self.bindings[name]
        else:
            if self.parent_frame is not None:
                return self.parent_frame.get_val(name)
            else:
                raise SchemeNameError("Name not in bindings or parent frame!")

    def set_val(self, name, val):
        self.bindings[name] = val

    def set_bang(self, var, exp):
        # print("frame bindings", self.bindings)
        if var in self.bindings:
            # print("var+bindings", var, self.bindings)
            self.bindings[var] = exp
        else:
            if self.parent_frame is not None:
                return self.parent_frame.set_bang(var, exp)
            else:
                raise SchemeNameError("Name not in bindings or parent frame!")


class Function:
    """
    Function class.
    """

    def __init__(self, params, expression, frame):
        self.body = expression
        self.params = params
        self.frame = frame

    def __call__(self, args):
        new_frame = Frame()
        new_frame.parent_frame = self.frame
        # bind the function's parameters to the arguments that are passed to it
        if len(args) != len(self.params) and self not in scheme_builtins.values():
            raise SchemeEvaluationError
        for i in range(len(self.params)):
            new_frame.set_val(self.params[i], args[i])

        return evaluate(self.body, new_frame)


if __name__ == "__main__":
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    # uncommenting the following line will run doctests from above
    # doctest.testmod()
    repl(True)
