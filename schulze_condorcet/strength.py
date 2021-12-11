"""Example implementations to define the strength of a link in the Schulze problem.

We view the candidates as vertices of a graph and determining the result as the
strongest path in the graph. To determine the strength of a path, we have to define a
metric which takes the voting of the voters into account.

Therefore, we transform every vote string into a number of supports and a number of
oppositions per path in the graph of candidates (for details, see at `schulze_evaluate`).
Now, we use this and the total number of votes to define the strength of a given path
in the graph of all candidates.

How to assess the strength of a path is one thing not specified by the Schulze method
and indeed there are several possibilities. We offer some sample implementations of such
a strength function, together with a protocol defining the interface of a generic
strength function (`StrenghtCallback` in module `schulze_condorcet.types`).
You can use it to implement your own strength function.
"""


def winning_votes(*, support: int, opposition: int, totalvotes: int) -> int:
    """This strategy is also advised by the paper of Markus Schulze.

    If two two links have more support than opposition, then the link with more
    supporters is stronger, if supporters tie then less opposition is used as secondary
    criterion.
    """
    if support > opposition:
        return totalvotes * support - opposition
    elif support == opposition:
        return 0
    else:
        return -1


def margin(*, support: int, opposition: int, totalvotes: int) -> int:
    """This strategy seems to have a more intuitive appeal.

    It sets the difference between support and opposition as strength of a link. However
    the discrepancy between this strategy and `winning_votes` is rather small, to wit
    all cases in the test suite give the same result for both of them.

    Moreover if the votes contain no ties both strategies (and several more) are totally
    equivalent.
    """
    return support - opposition
